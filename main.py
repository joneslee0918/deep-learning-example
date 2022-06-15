import sys, traceback
from flask import Flask, send_from_directory, request
import random
import string
import cv2
import glob
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
_PATH_ = os.path.dirname(os.path.realpath(__file__))


def randString(len = 5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(len))

def joinPath(*paths):
    tmpPath = os.path.join(*paths)
    new_path = os.path.dirname(tmpPath)
    
    print(new_path, os.path.isfile(new_path), os.path.isdir(new_path))
    os.makedirs(new_path, exist_ok=True)
    os.chmod(new_path , 777)
    return tmpPath

def FrameCapture(video_path, output_path):
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            image = cv2.resize(image, (216, 384)) 
            cv2.imwrite(output_path + "/%d.jpg" % count, image)
            count += 1

def videoExtract(clips_path, videopath):
    frame_path = os.path.join(clips_path)
    os.makedirs(frame_path)
    FrameCapture(videopath, frame_path)

def colorize_video(input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4
    image_size = [216 *2, 384 *2]
    frame_propagate = False

    # processing folders
    mkdir_if_not(output_path)
    files = glob.glob(output_path + "*")
    print("processing the folder:", input_path)
    path, dirs, filenames = os.walk(input_path).__next__()
    file_count = len(filenames)
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

    # NOTE: resize frames to 216*384
    transform = transforms.Compose(
        [CenterPad(image_size), transform_lib.CenterCrop(image_size), RGB2Lab(), ToTensor(), Normalize()]
    )
    

    # if frame propagation: use the first frame as reference
    # otherwise, use the specified reference image
    ref_name = input_path + filenames[0] if frame_propagate else reference_file
    print("reference name:", ref_name)
    print("check => -1")
    frame_ref = Image.open(ref_name)
    print("check => 0")

    total_time = 0
    I_last_lab_predict = None

    IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
    print("check => 1")
    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
    IB_l = IB_lab[:, 0:1, :, :]
    IB_ab = IB_lab[:, 1:3, :, :]
    print("check => 2")
    with torch.no_grad():
      I_reference_lab = IB_lab
      I_reference_l = I_reference_lab[:, 0:1, :, :]
      I_reference_ab = I_reference_lab[:, 1:3, :, :]
      I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
      features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
    print("check => 3")

    for index, frame_name in enumerate(tqdm(filenames)):
        print("check => 4")
        frame1 = Image.open(os.path.join(input_path, frame_name))
        IA_lab_large = transform(frame1).unsqueeze(0).cuda()
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")

        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]
        
        if I_last_lab_predict is None:
            if frame_propagate:
                I_last_lab_predict = IB_lab
            else:
                I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

        # start the frame colorization
        print("check => 5")
        with torch.no_grad():
            I_current_lab = IA_lab
            I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                I_current_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        # upsampling
        print("check => 6")
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
        )

        # filtering
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])
        print("check => 7")

        # save the frames
        save_frames(IA_predict_rgb, output_path, index)

    # output video
    video_name = "video.avi"
    print("output_path => ", output_path, video_name)
    folder2vid(image_folder=output_path, output_dir=output_path, filename=video_name)
    print()

def colorize(clip_path, ref_path, output_path):
    cudnn.benchmark = True
    clip_name = clip_path.split("/")[-1]
    refs = os.listdir(ref_path)
    refs.sort()

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()

    for ref_name in refs:
        try:
            colorize_video(
                clip_path,
                joinPath(ref_path, ref_name),
                joinPath(output_path, clip_name + "_" + ref_name.split(".")[0]),
                nonlocal_net,
                colornet,
                vggnet,
            )
        except Exception as error:
            print("error when colorizing the video " + ref_name)
            traceback.print_exc()

    video_name = "video.avi"
    clip_output_path = joinPath(output_path, clip_name)
    mkdir_if_not(clip_output_path)
    folder2vid(image_folder=clip_path, output_dir=clip_output_path, filename=video_name)
    pass

app = Flask(__name__)
cf_port = 80

@app.route("/")
def hello_world():
    return send_from_directory(_PATH_, 'index.html')

@app.route("/colorizing", methods=['POST'])
def colorizing():
    v_file = request.files['video']
    r_files = request.files.getlist('ref[]')

    uploadPath = joinPath(_PATH_, "example")
    clip_path = joinPath(uploadPath, "clips")
    output_path = joinPath(uploadPath, 'output')
    ref_path = joinPath(uploadPath, 'refs')

    filename = randString() + v_file.filename
    vpath = joinPath(uploadPath, filename)
    v_file.save(vpath)

    videoExtract(clip_path, vpath)

    for file in r_files:
        filename = randString() + file.filename
        file.save(joinPath(ref_path, filename))
    
    result = colorize(clip_path, ref_path, output_path)


    return "<a href='../'>back</a> <br/>\
         <a href='%s' download> result</a>   \
         <h3>Success</h3/>" % result
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(cf_port), debug=True)
