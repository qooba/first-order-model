import os
import argparse
import gdown
import imageio
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from skimage.transform import resize
#from IPython.display import HTML
import warnings
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte
from image_crop import image_crop
warnings.filterwarnings("ignore")


class Runner:
    def run(self, opt):
        if opt.crop_image:
            source_image = image_crop(opt.source_image, opt.crop_image_padding)
        else:
            source_image = imageio.imread(opt.source_image)

        reader = imageio.get_reader(opt.driving_video)
        source_image = resize(source_image, (256, 256))[..., :3]
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
        imageio.mimsave(opt.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='first-order-model')
    parser.add_argument("--config", default='./config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='./checkpoints/vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument('--source_image', default='./test/02.png', type=str, help='source image')
    parser.add_argument('--driving_video', default='./test/04.mp4', type=str, help='driving video')
    parser.add_argument('--crop_image', '-ci', action='store_true', help='autocrop image')
    parser.add_argument('--crop_image_padding', '-cip', nargs='+', type=int, help='autocrop image paddings left, upper, right, lower')
    parser.add_argument('--crop_video', '-cv', action='store_true', help='autocrop video')
    parser.add_argument('--output', default='./generated.mp4', type=str, help='output video')

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    opt = parser.parse_args()

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints",exist_ok=True)
        gdown.download("https://drive.google.com/u/0/uc?id=1L8P-hpBhZi8Q_1vP2KlQ4N6dvlzpYBvZ","./checkpoints/vox-adv-cpk.pth.tar")
        gdown.download("https://drive.google.com/u/0/uc?id=1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS","./checkpoints/vox-cpk.pth.tar")



    if not os.path.exists("test"):
        os.makedirs("test",exist_ok=True)
        gdown.download("https://drive.google.com/u/0/uc?id=1b8dEqviM-UvMlKh2oKnWNMDGTsAjuAgW","./test/02.png")
        gdown.download("https://drive.google.com/u/0/uc?id=1-3e_CgUnB7WUIdoSEGatv7klPfk8pRc0","./test/04.mp4")


    Runner().run(opt)
