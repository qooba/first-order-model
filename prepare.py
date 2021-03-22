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
from demo import load_checkpoints, make_animation, find_best_frame
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

        if opt.find_best_frame or opt.best_frame is not None:
            i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

        #predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
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
    parser.add_argument("--no-relative", dest="relative", action="store_false", help="don't use relative or absolute keypoint coordinates")
    parser.set_defaults(relative=True)
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--no-adapt_scale", dest="adapt_scale", action="store_false", help="no adapt movement scale based on convex hull of keypoints")
    parser.set_defaults(adapt_scale=True)

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    opt = parser.parse_args()
    for key, val in vars(opt).items():
        if isinstance(val, list):
            val = [str(v) for v in val]
            val = ','.join(val)
        if val is None:
            val = 'None'
        print('{:>20} : {:<50}'.format(key, val))

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints",exist_ok=True)
        gdown.download("https://drive.google.com/u/0/uc?id=1L8P-hpBhZi8Q_1vP2KlQ4N6dvlzpYBvZ","./checkpoints/vox-adv-cpk.pth.tar")
        gdown.download("https://drive.google.com/u/0/uc?id=1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS","./checkpoints/vox-cpk.pth.tar")



    if not os.path.exists("test"):
        os.makedirs("test",exist_ok=True)
        gdown.download("https://drive.google.com/u/0/uc?id=1b8dEqviM-UvMlKh2oKnWNMDGTsAjuAgW","./test/02.png")
        gdown.download("https://drive.google.com/u/0/uc?id=1-3e_CgUnB7WUIdoSEGatv7klPfk8pRc0","./test/04.mp4")


    Runner().run(opt)
