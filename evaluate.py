
from heatblob import infer_sequences

def evaluate_on_folder(folder=None, scale=1.5):
    os.system('rm t_imgs/*.jpg')
    if folder is None: folder = plate_crops_folder
    filenames = glob(folder+'/*.jpg')
    for i, filename in enumerate(filenames[:100]):
        img = cv2.imread(filename, 0)
        if img is None: continue
        img = get_scaled_crop(img)
        img1 = img
        img2 = 255 - img1
        outputs = pass_imgs(np.asarray([img1,img2]))
        output1 = outputs[0]
        output2 = outputs[1]
        output_pair(torch.from_numpy(img1.transpose([2,0,1])), output1, 't_imgs', i*2+1)
        output_pair(torch.from_numpy(img2.transpose([2,0,1])), output2, 't_imgs', i*2+2)


def test2(idx=0, shows=[]):
    global model, filenames, filename, img, heatmaps, background
    if model is None: model = torch.load('model2.pth')
    filenames = glob(plate_crops_folder+'/*.jpg')
    filename = filenames[idx]
    img = cv2.imread(filename, 0)
    img = get_scaled_crop(img)
    if 0 in shows:
        print(img.shape, img.dtype)
        plt.imshow(img[:,:,0])
        plt.show()
    heatmaps = pass_imgs(np.expand_dims(img, 0)).numpy()
    sequences, filtered = infer_sequences(heatmaps[0], shows)
    print("Raw sequences")
    pprint(sequences)
    print("Filtered sequences")
    pprint(filtered)

def test2_group():
    global model, filenames, filename, img, heatmaps, background
    if model is None: model = torch.load('model2.pth')
    filenames = glob(plate_crops_folder+'/*.jpg')[:len(test2_labels)]
    num_fail = 0
    for i, filename in enumerate(filenames):
        label = test2_labels[i]
        img = cv2.imread(filename, 0).astype(np.float32)
        img = img - img.min()
        img = img / img.max() * 255
        img = img.astype(np.uint8)
        img = get_scaled_crop(img)
        heatmaps = pass_imgs(np.expand_dims(img, 0)).numpy()
        sequences, filtered = infer_sequences(heatmaps[0])
        if len(filtered)==0:
            print(i, label.ljust(8), 'empty', sequences[0].get_str_seq())
            #print("Error {}, label: {}, #####Top answer: {}".format(i, label, sequences[0].get_str_seq()))
            num_fail += 1
        elif filtered[0].get_str_seq() != label:
            print(i, label.ljust(8), 'wrong', filtered[0].get_str_seq())
            #print("Error {}, label: {}, Filtered answer: {}".format(i, label, filtered[0].get_str_seq()))
            num_fail += 1
    print('Number of failures: {}'.format(num_fail))

test2_labels = [
        'SJP7348Z',
        'SKQ614J', 
        'EM788H',   
        'SCN812P', 
        'SLE4770R',
        'SHA8740U',
        'SJX8017E',
        'SLH145G',
        'SKU1310H',
        'SLF2199Z',
        'SLF9712X',
        'SJU1022B',
        'SHD4345M',
        'SKK8076R',
        'SLC295H',
        'SHC5370M',
        'SHC1312H',
        'SLJ9211D',
        'SHC7878C',
        'SLF6100G',
        'SHB6260X',
        'GU2800B',
        'SLB4381X',
        'SJN3169K',
        'SHB1798U',
        'GQ8883T',
        'SLJ3990C',
        'SHD7140S',
        'EY823B',
        'GBD6960U',
        'SJJ7899G', 
        'SLL4070R',
        'GV1229C',
        'SLD9148K',
        'SHC8901L',
        'SJQ9175L',
        'SDU2333T',
        'SKW5741K', 
        'GU1405R',
        'SFL2223B',
        'SLE2716L',
        'SCJ6188P',
        'SHD3193R',
        'SJR7233K',
        'SLJ5305T',
        'SLK4762H',
        'SHA1655D',
        'SJG7462Z',
        'SHC6931M',
        'SH3918R', 
        'SGW8066D',
        'SJT446Z',
        'SJX7428J',
        'SH8297R', 
        'EZ5115Z', 
        'SHA1215R',
        'SGY219Y',
        'SFT2604T',
        'SDD688Y', 
        'SKK51P',
        'SGU9478C',
        'SGD2848D',
        'SJQ5604S',
        'SHB4507Z',
        'SKV2715L',
        'SLA1715S',
        'SKF7206R',
        'SKU858B',
        'YP4189H', 
        'SHD8572Y',
        'SGG8761P',
        'SKX4678L',
        'SHC2178Y',
        'SLC3362C',
        'SLJ9639D',
        'SHB5813H',
        'SLK4171M',
        'SJB2002C',
        'SHC7524Z',
        'GZ6199B', 
        'SDS6515U',
        'SJB8387C',
        'SJV1249J',
        'SLL3136S',
        'SLB6705H',
        'SHC1585L',
        'SHD3562J',
        'SHC122X', 
        'GU2499M', 
        'YN6727C',
        'SKJ1227Z',
        'SLJ5217P',
        'GX9508J',
        'SFT1701C',
        'SKD3628T',
        'SHC4665U',
        'SLK9918T',
        'YL8955H',
        'SKQ5961Z',
        'SKG85M',
]
