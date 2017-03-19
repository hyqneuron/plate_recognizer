import math
import numpy as np
import itertools
from pprint import pprint
import cv2
from plate_gen.plate import charid_to_char
import matplotlib.pyplot as plt
#from main import num_classes, num_foreground
num_foreground = 37
num_classes = num_foreground + 1

debug_items = []
class HeatBlob:
    def __init__(self, centerx, centery, w, h, blobmask, maskedheat, char=None, fore_blobs=None):
        self.centerx = centerx
        self.centery = centery
        self.w = w
        self.h = h
        self.blobmask = blobmask        # 1s covering the region of the blob
        self.maskedheat = maskedheat    # blobmask * back_heat
        self.area = blobmask.sum()
        self.totalheat = maskedheat.sum()
        self.char = char # is None for background blobs, otherwise is the corresponding character
        if fore_blobs is None: fore_blobs = []
        self.fore_blobs = fore_blobs
        self.distribution = {} # maps character to (probability, blob)
    def compute_distribution(self, heatmaps):
        """
        self has to be a background blob
        Given the fore_blobs associated with this back_blob, compute the probability of each fore_blob using their
        totalheat
        """
        assert self.char is None, 'This func only works on background blobs'
        """
        heatsum = sum([b.totalheat for b in self.fore_blobs])
        self.distribution = {}
        for fore_blob in self.fore_blobs:
            assert fore_blob.char is not None
            self.distribution[fore_blob.char] = (fore_blob.totalheat / heatsum, fore_blob)
        """
        assert len(self.fore_blobs)==0
        self.fore_blobs = []
        # compute total heat to make sure things make sense
        char_heat = [0] * num_foreground
        for charid in range(num_foreground):
            char_heat[charid] = (heatmaps[charid] * self.blobmask).sum()
        totalheat = sum(char_heat)
        assert abs(totalheat - self.totalheat) < 0.0001
        # remove those blobs with less than 1% probability
        for charid in range(num_foreground):
            if char_heat[charid] < totalheat * 0.10:
                char_heat[charid] = 0
        totalheat = sum(char_heat)
        # generate fore_blobs
        for charid in range(num_foreground):
            if char_heat[charid] > 0:
                char = charid_to_char(charid)
                fore_blob = HeatBlob(self.centerx, self.centery, self.w, self.h, 
                                     self.blobmask, self.blobmask * heatmaps[charid], char=char)
                self.fore_blobs.append(fore_blob)
                self.distribution[char] = (char_heat[charid] / totalheat, fore_blob)
    def __repr__(self):
        return str(self.char)

class PlateSequence:
    def __init__(self, blobs, prob):
        self.blobs = blobs
        self.prob = prob
    def get_str_seq(self):
        return ''.join([b.char for b in self.blobs])
    def __str__(self):
        return str((self.get_str_seq(), self.prob))
    def __repr__(self):
        return str(self)


def get_blobs_from_heatmap(heatmap, char, threshold):
    global debug_items
    """
    heatmap: HxW
    From a heatmap, extract the heat blobs that exceed threshold
    Return a list of HeatBlob(centerx, centery, w, h, blobmask, maskedheat)
    """
    H,W = heatmap.shape[:2]
    threshed = heatmap > threshold
    contours, _ = cv2.findContours(threshed.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        blobmask = np.zeros((H,W), np.float32)
        cv2.fillPoly(blobmask, [cnt], 1.0)
        x,y,w,h = cv2.boundingRect(cnt)
        centerx = x + w / 2
        centery = y + h / 2
        if blobmask.sum() < 4: continue # skip blob if it's too small
        """ sometimes 2 blobs are joined together, detect and handle that"""
        if 'wh' in debug_items and char in debug_items:
            print(w, h)
        aspect_ratio = w / float(h)
        if (char=='1' and aspect_ratio >= 1.5) or (char!='1' and aspect_ratio >= 2.0):
            centerx1 = int(centerx - w * 0.25)
            centerx2 = int(centerx + w * 0.25)
            blobmask_selector1 = np.zeros((H,W), np.float32)
            blobmask_selector2 = np.zeros((H,W), np.float32)
            blobmask_selector1[:, :centerx] = 1
            blobmask_selector2[:, centerx:] = 1
            blobmask1 = blobmask * blobmask_selector1
            blobmask2 = blobmask * blobmask_selector2
            maskedheat1 = blobmask1 * heatmap
            maskedheat2 = blobmask2 * heatmap
            if blobmask1.sum() >= 4:
                blobs.append(HeatBlob(centerx1, centery, w/2, h, blobmask1, maskedheat1, char))
            if blobmask2.sum() >= 4:
                blobs.append(HeatBlob(centerx2, centery, w/2, h, blobmask2, maskedheat2, char))
        else:
            maskedheat = blobmask * heatmap
            blobs.append(HeatBlob(centerx, centery, w, h, blobmask, maskedheat, char))
    if char in debug_items:
        plt.imshow(threshed)
        plt.show()
    return blobs

def pre_sort_blobs(blobs):
    """ 
    sort the blobs in the order of top-to-bottom, then left-to-right 
    One possible way is to sort by blob.centerx * blob.centery, but centery*centery*centerx is more robust
    """
    return sorted(blobs, key=lambda b:b.centery*b.centery*b.centerx)

def sort_blobs_by_x(blobs):
    return sorted(blobs, key=lambda b:b.centerx)

def find_blob_peers(blobs, startidx, ythresh = 4):
    """
    starting from startidx, find the subsequent blobs (inclusive) that are on the same line as the startidx blob 
    """
    if startidx >= len(blobs): return []
    prev_y = blobs[startidx].centery
    peers = []
    for idx in xrange(startidx, len(blobs)):
        blob = blobs[idx]
        if abs(prev_y - blob.centery) <= ythresh:
            peers.append(blob)
            prev_y = blob.centery
        else:
            break
    return peers

def arrange_blobs_by_lines(blobs):
    global debug_items
    """
    arrange the blobs by lines
    Return a list of Line, a Line is a list of HeatBlob
    """
    startidx = 0
    lines = []
    while startidx < len(blobs):
        peers = find_blob_peers(blobs, startidx)
        assert len(peers) > 0, 'cannot ever have 0 peers, cos peers include itself'
        if len(peers) == 1: # no line!
            # just skip this blob
            if 'warn' in debug_items:
                print('Warning: {}th blob is isolated'.format(startidx))
        else: # minimum 2 peers
            lines.append(peers)
        startidx += len(peers)
    return lines

def associate_fore_back(fore_blob, back_blobs, thresh=3):
    global debug_items
    """
    Find the background blob that the fore_groundblob is closest to. If the distance is below threshold, add the
    foreground_blob to the corresponding background blob's fore_blobs
    """
    dist_min = 99999
    blob_min = None
    for back_blob in back_blobs:
        fx,fy = fore_blob.centerx, fore_blob.centery
        bx,by = back_blob.centerx, back_blob.centery
        dist = math.sqrt((fx - bx)**2 + (fy - by)**2)
        if dist < dist_min:
            dist_min = dist
            blob_min = back_blob
    if dist_min <= thresh:
        blob_min.fore_blobs.append(fore_blob)
        return True
    else:
        if 'warn' in debug_items:
            print('Association failure', dist_min, thresh)
        return False

def infer_sequences(heatmaps, shows=[]):
    global debug_items
    debug_items = shows
    """
    heatmaps: num_classes by H by W
    Infer the number sequence from the heats
    1. Extract the background blobs, sort by order or top-to-bottom, left-to-right
    2. Arrange background blobs by lines
    3. Find blobs for all foreground heatmaps
    4. Associate foreground heat blobs with background heatblobs
    5. for each back_blob, compute a distribution over its associated fore_blobs
    6. Pull out the raw sequences
    7. Postprocess sequences
    """
    """ get the character blobs from background """
    background = 1 - heatmaps[num_classes-1]
    if 1 in debug_items:
        plt.imshow(background)
        plt.show()
    back_blobs = get_blobs_from_heatmap(background, None, 0.3)
    """ 1-line or 2-line? """
    back_blobs = pre_sort_blobs(back_blobs)
    lines = arrange_blobs_by_lines(back_blobs)
    for i,line in enumerate(lines):
        lines[i] = sort_blobs_by_x(line)
    back_blobs = sum(lines, [])

    if 'info' in debug_items:
        print('number of lines: {}, lengths: {}'.format(len(lines), map(len, lines)))
    """
    # Find blobs for all foreground heatmaps 
    char_heatblobs = {}
    for charid in xrange(num_foreground):
        char = charid_to_char(charid)
        heatblobs = get_blobs_from_heatmap(heatmaps[charid], char, 0.1)
        char_heatblobs[char] = heatblobs
    # Associate foreground blobs with background blobs
    for char, heatblobs in char_heatblobs.iteritems():
        if len(heatblobs) == 0: continue
        for i, fore_blob in enumerate(heatblobs):
            associate_success = associate_fore_back(fore_blob, back_blobs)
            if not associate_success and 'warn' in debug_items:
                print("Warning: {}th blob for {} was not associated".format(i, char))
    """
    # for each back_blob, compute a distribution over its associated fore_blobs 
    for back_blob in back_blobs:
        back_blob.compute_distribution(heatmaps)
        if 'info' in debug_items:
            pprint(back_blob.distribution)
    """ Pull out the raw sequences """
    if sum([len(line) for line in lines]) > 13: # some crazy shit have so many blobs it explodes here
        return [], []
    sequences = get_raw_sequences_for_lines(lines)
    sequences = sorted(sequences, reverse=True, key=lambda seq:seq.prob) # sort by probability in descending order
    sequences = sequences[:30] # only take top 30
    """ postprocess """
    postprocessed_sequences = postprocess_sequences(sequences)
    return sequences, postprocessed_sequences

def get_raw_sequences_for_line(line):
    """
    line is a list of HeatBlob
    extract set of all possible sequences from this line
    """
    sequences = [PlateSequence([], 1.0)]
    for back_blob in line:
        new_sequences = []
        for char, (prob, fore_blob) in back_blob.distribution.iteritems():
            new_sequences += [PlateSequence(seq.blobs+[fore_blob], seq.prob*prob) for seq in sequences]
        if len(new_sequences) > 0: 
            sequences = new_sequences
    if 'info' in debug_items:
        print("Showing line")
        pprint(sequences)
    return sequences


def get_raw_sequences_for_lines(lines):
    """
    Simply list out all possible sequences
    """
    sequences = [PlateSequence([], 1)]
    for line in lines:
        line_sequences = get_raw_sequences_for_line(line)
        new_sequences = []
        for seq1, seq2 in itertools.product(sequences, line_sequences):
            new_sequences.append(PlateSequence(seq1.blobs+seq2.blobs, seq1.prob*seq2.prob))
        if len(new_sequences) > 0: 
            sequences = new_sequences
    return sequences

def postprocess_sequences(sequences):
    global debug_items
    """ 
    1. Run checksum filter. If non-empty, just return
    2. Diagnose headword, numword, and sumseq
    """
    if len(sequences)==0: 
        return sequences
    """ try checksum """
    cached_results = {}
    results = sum(map(lambda s:fix_sequence(s, 0, cached_results), sequences), [])
    return sorted(results, reverse=True, key=lambda seq:seq.prob)

def fix_sequence(seq, depth, cached_results):
    global debug_items
    """
    seq is a problematic sequence rejected by has_valid_checksum
    provide a list of sequences, each of which being a 'fixed' version of seq
    """
    str_seq = seq.get_str_seq()

    if len(seq.blobs) > 12: return []

    if len(cached_results) > 1000:
        return []

    def cr(results):
        cached_results[str_seq] = results
        return results

    if str_seq in cached_results:
        return cached_results[str_seq]
    cached_results[str_seq] = [] # prevent other threads from entering this sequence

    if depth >= 8:
        print(depth, str_seq)
        return cr([]) # giveup
    if 'debug' in debug_items:
        print('fix_sequence', seq, cached_results)
    if has_valid_checksum(seq):
        """
        If it has valid checksum, just return
        But before we return, make sure this thing doesn't have a headseq of length 3, yet still begin with something
        that isn't S. If it begins with something isn't S/G, we remove the head.
        """
        result = split_seq(seq, ignore_length=True)
        assert type(result)==tuple
        (_,_,_), (headseq, numseq, checkseq) = result
        if len(headseq)==3 and headseq[0].char not in ['S', 'G']:
            new_seq = PlateSequence(seq.blobs[1:], seq.prob)
            return cr([new_seq])
        else:
            return cr([seq])
    while True:
        result = split_seq(seq)
        if type(result) != str:
            return cr([]) # can't fix it, cos it's a checksum error
        if 'debug' in debug_items:
            print(result)
        if result in ['seq_too_short', 'headword_too_short', 'numword_too_short']:
            return cr([]) # can't fix it, so just return empty
        if result=='head_is_number':
            # drop head
            new_seq = PlateSequence(seq.blobs[1:], seq.prob) # FIXME correct prob
            return cr(fix_sequence(new_seq, depth+1, cached_results))
        elif result in ['second_tail_is_letter', 'tail_is_number']:
            # drop tail
            new_seq = PlateSequence(seq.blobs[:-1], seq.prob) # FIXME correct prob
            return cr(fix_sequence(new_seq, depth+1, cached_results))
        elif result=='headword_too_long':
            # drop either side of headword
            (_,_,_), (headseq, numseq, checkseq) = split_seq(seq, ignore_length=True)
            new_seq1 = PlateSequence(headseq[1:] +numseq+checkseq, seq.prob) # FIXME correct prob
            new_seq2 = PlateSequence(headseq[:-1]+numseq+checkseq, seq.prob) # FIXME correct prob
            return cr(fix_sequence(new_seq1, depth+1, cached_results)+fix_sequence(new_seq2, depth+1, cached_results))
        elif result=='numword_too_long':
            # drop either side of numword
            (_,_,_), (headseq, numseq, checkseq) = split_seq(seq, ignore_length=True)
            new_seq1 = PlateSequence(headseq+numseq[1:] +checkseq, seq.prob) # FIXME correct prob
            new_seq2 = PlateSequence(headseq+numseq[:-1]+checkseq, seq.prob) # FIXME correct prob
            return cr(fix_sequence(new_seq1, depth+1, cached_results)+fix_sequence(new_seq2, depth+1, cached_results))
        elif result=='headword_has_number':
            # drop all numbers from headword
            (_,_,_), (headseq, numseq, checkseq) = split_seq(seq, ignore_length=True)
            headseq = filter(lambda blob:is_letter(blob.char), headseq)
            new_seq = PlateSequence(headseq+numseq+checkseq, seq.prob) # FIXME correct prob
            return cr(fix_sequence(new_seq, depth+1, cached_results))
        else:
            assert False, 'result is an unknown error message from split_seq'


""" 
======================================= checksum related stuffs ===========================================
"""
def is_letter(char):
    return ord('A') <= ord(char) <= ord('Z')

def is_number(char):
    return ord('0') <= ord(char) <= ord('9')

def headword_to_numbers(headword):
    """ Turns the 1/2/3 letters at the beginning of the plate into 2 numbers """
    while len(headword) < 2:
        headword = '@' + headword
    if len(headword) == 3: headword = headword[1:]
    return map(lambda c:ord(c)-64, headword)

def numword_to_numbers(numword):
    """ Turns the 1/2/3/4 numbers at the beginning of the plate into 4 numbers """
    while len(numword) < 4:
        numword = '0' + numword
    return map(lambda c:ord(c)-48, numword)

def compute_checksum(headword, numword):
    """
    Turns headword and numword into a checksum character
    Computation steps documented in Wikipedia's 'Vehicle registration plates of Singapore' page
    """
    numbers = headword_to_numbers(headword) + numword_to_numbers(numword)
    result = (np.asarray(numbers) * np.asarray([9, 4, 5, 4, 3, 2])).sum()
    choices = ('A', 'Z', 'Y', 'X', 'U', 'T', 'S', 'R', 'P', 'M', 'L', 'K', 'J', 'H', 'G', 'E', 'D', 'C', 'B')
    return choices[result % len(choices)]

def split_seq(seq, ignore_length=False):
    """
    Split sequence into headword, numword, checksum
    eg: SLA1234K: SLA is headword, 1234 is numword, K is checksum
    If splitting cannot succeed, return reason of failure
    """
    if 'debug' in debug_items:
        print('split_seq', seq)
    word = seq.get_str_seq()
    if len(word) < 4: return 'seq_too_short'                     # solution: none
    if not is_letter(word[0]): return 'head_is_number'           # solution: drop head
    if not is_letter(word[-1]): return 'tail_is_number'          # solution: none
    if not is_number(word[-2]): return 'second_tail_is_letter'   # solution: drop tail
    i = len(word) - 1
    while True:
        i -= 1
        if i < 0: break
        if not is_number(word[i]): break
    headword = word[:i+1]
    numword = word[i+1:-1]
    checksum = word[-1]
    if ignore_length:
        return (headword, numword, checksum), (seq.blobs[:i+1], seq.blobs[i+1:-1], [seq.blobs[-1]])
    if not all(map(is_letter, headword)): return 'headword_has_number'
    # if not all(map(is_number, numword)) return 'numword_has_number'
    # above isn't possible, because we explicitly make sure numword contains numbers only
    if len(headword) > 3: return 'headword_too_long'              # solution: drop sides of headword
    if len(headword) < 1: return 'headword_too_short'             # solution: none
    if len(numword)  > 4: return 'numword_too_long'               # solution: drop one of numword
    if len(numword)  < 1: return 'numword_too_short'              # solution: none
    return headword, numword, checksum

def has_valid_checksum(seq):
    """
    seq is a string
    check whether seq constitutes valid Singaporean plate number
    """
    result = split_seq(seq)
    if type(result)==str:
        return False
    headword, numword, checksum = result
    return compute_checksum(headword, numword)==checksum

"""
todos:
- change back-blob based distribution computation to remove foreground thresholding, and use 
  backblob.heatmask * foreground[charid] instead
- proper line filtering
- proper line detection
* filter out bad checksums
* use checksum logic to fixup plates when everything gets filtered out
"""
if __name__=='__main__':
    print(has_valid_checksum('SDS6515U'))


