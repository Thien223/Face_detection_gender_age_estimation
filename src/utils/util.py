import torch
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def annotating_coco_dataset():
    from facenet_pytorch import MTCNN
    import matplotlib.image as mpl
    import matplotlib.pyplot as plt
    from cv2 import cv2
    def convert_to_darknet(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    mtcnn = MTCNN()

    path = r'.\coco_images'
    labels_path = r'.\coco_labels'
    ims = os.listdir(path)
    ignored_ones = []

    for img in ims:

        im_path = path + '\\' + img
        im = mpl.imread(im_path)

        try:
            boxes, probs, points = mtcnn.detect(im, landmarks=True)
        except RuntimeError as e:
            print(f"Couldn't detect {im_path}")
            continue

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                startX, startY, endX, endY = box.astype(int)

                color = (0, 255, 0)
                cv2.putText(im,
                            f'{prob:.1%}',
                            (startX, startY - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.5,
                            color=color,
                            thickness=2)
                cv2.rectangle(im, (startX, startY), (endX, endY), color, 2)

            w = int(im.shape[0])
            h = int(im.shape[1])

            label = img.rstrip('.jpg')
            with open(rf'{labels_path}\{label}.txt', 'w') as f:
                for item in boxes:
                    b = (startX, endX, startY, endY)
                    # convert_to_darknet at
                    # https://gist.github.com/AlexanderNixon/fb741fa2e950c7e0228394027ff9dffc
                    bb = convert_to_darknet((w, h), b)
                    box = ' '.join(item.astype(str))
                    f.write(f"0 {box}\n")
            print(img)
            plt.imshow(im)
            plt.show()

        else:
            ignored_ones.append(im_path)


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def load_checkpoint(model_path):
    if model_path is None:
        model_path = 'models/checkpoints/vgg-epochs_310-step_0-gender_acc_0.9651533961296082-age_acc_0.7990920543670654.pth'
    # model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint['model_type']
    epoch = checkpoint['parameter']['epoch']
    m = None
    if model_type == 'vgg':
        from modules.vgg import VGG
        m = VGG(vgg_type='vgg19')
    elif model_type == 'cspvgg':
        from modules.vgg import CSP_VGG
        m = CSP_VGG(vgg_type='vgg19')
    elif model_type == 'inception':
        from modules.vgg import Inception_VGG
        m = Inception_VGG(vgg_type='vgg19')
    m.load_state_dict(checkpoint['model_state_dict'])
    return m, epoch

#
# if __name__ == '__main__':
# 	import sys
# 	import getopt
# 	from glob import glob
#
# 	args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size='])
# 	args = dict(args)
# 	args.setdefault('--debug', './output/')
# 	args.setdefault('--square_size', 1.0)
# 	if not img_mask:
# 		img_mask = 'templates/left*.jpg'  # default
# 	else:
# 		img_mask = img_mask[0]
#
# 	img_names = glob(img_mask)
# 	debug_dir = args.get('--debug')
# 	if not os.path.isdir(debug_dir):
# 		os.mkdir(debug_dir)
# 	square_size = float(args.get('--square_size'))
#
# 	pattern_size = (9, 6)
# 	pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
# 	pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
# 	pattern_points *= square_size
#
# 	obj_points = []
# 	img_points = []
# 	h, w = 0, 0
# 	img_names_undistort = []
# 	for fn in img_names:
# 		print('processing %s... ' % fn, end='')
# 		img = cv2.imread(fn, 0)
# 		if img is None:
# 			print("Failed to load", fn)
# 			continue
#
# 		h, w = img.shape[:2]
# 		found, corners = cv2.findChessboardCorners(img, pattern_size)
# 		if found:
# 			term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
# 			cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
#
# 		if debug_dir:
# 			vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 			cv2.drawChessboardCorners(vis, pattern_size, corners, found)
# 			path, name, ext = splitfn(fn)
# 			outfile = debug_dir + name + '_chess.png'
# 			cv2.imwrite(outfile, vis)
# 			if found:
# 				img_names_undistort.append(outfile)
#
# 		if not found:
# 			print('chessboard not found')
# 			continue
#
# 		img_points.append(corners.reshape(-1, 2))
# 		obj_points.append(pattern_points)
#
# 		print('ok')
#
# 	# calculate camera distortion
# 	rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
#
# 	mean_error = 0
# 	for i in range(len(obj_points)):
# 		imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
# 		error = cv2.norm(img_points[i], imgpoints2.reshape(img_points[i].shape), cv2.NORM_L2) / len(imgpoints2)
# 		mean_error += error
#
# 	print("Projection error: (smaller is better) ", mean_error / len(obj_points))
# 	print("\nRMS:", rms)
# 	print("camera matrix:\n", camera_matrix)
# 	print("distortion coefficients: ", dist_coefs.ravel())
#
# 	# undistort the image with the calibration
# 	print('')
# 	for img_found in img_names_undistort:
# 		img = cv2.imread(img_found)
#
# 		h,  w = img.shape[:2]
# 		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
#
# 		dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
#
# 		# crop and save the image
# 		x, y, w, h = roi
# 		dst = dst[y:y+h, x:x+w]
# 		outfile = img_found + '_undistorted.png'
# 		print('Undistorted image written to: %s' % outfile)
# 		cv2.imwrite(outfile, dst)
#
# 	cv2.destroyAllWindows()
