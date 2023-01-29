import numpy as np
import cv2

from planarH import compositeH, computeH_ransac, orderMatches
from matchPics import matchPics
from displayMatch import displayMatched
from opts import get_opts
from cpselect.cpselect import cpselect


def rescaleImg(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def intP2coords(interestPoints: dict):
    """ Transforms the image coordantes mapped using cpselect to x,y locations for image 1 and image 2.

    Args:
        interestPoints (dict): dict returned from cpselect, containing all the manually selected matches from image 1 and image 2.

    Returns:
        x1: np.array of the x, y locations coming from image 1
        x2: np.array of the x, y locations coming from image 2
    """

    # ? ----------------- Steps ------------------------------
    """
    
    1. Establish how many interest points were selected
    2. generate x1, x2 points of length = # interest points
    3. [For i interest points] extract x, y points and save them to x1, x2
    
    """
    # 1
    N = len(interestPoints)

    # 2
    x1 = np.zeros(shape=(N, 2))
    x2 = np.zeros(shape=(N, 2))

    # 3
    for i in range(N):

        x1[i, 0], x1[i, 1] = interestPoints[i]["img1_x"], interestPoints[i]["img1_y"]

        x2[i, 0], x2[i, 1] = interestPoints[i]["img2_x"], interestPoints[i]["img2_y"]

    return x1, x2


def panorama(image1_dest, image2_dest, show_matches=False):
    """ Given the file name of two images, generate and display a panorama.

    Args:
        image1_dest (str): File name, must relative to folder as follows: '../data/image1_dest')
        image2_dest (str): File name, must relative to folder as follows: '../data/image1_dest')
    """

    # ? ----------------- Steps -----------------------------

    """
    1. Open the images
    2. Find the interest points manually 
    3. Change the dict to two np.arrays that work with functions from planarH
    4. Find the homography
    5. Apply the homography to image 2
    6. Rescale the images to reasonable sizes (15% of original)
    7. Replace all the black pixels with image 1
    8. Show the panorama
    
    """

    # import opts
    opts = get_opts()

    # 1
    image1 = cv2.imread(filename="../data/" + image1_dest)
    image2 = cv2.imread(filename="../data/" + image2_dest)

    # 2
    interestPoints = cpselect(
        img_path1="../data/" + image1_dest, img_path2="../data/" + image2_dest)

    #! good interest points from one trial run
    # interestPoints = [{'point_id': 1, 'img1_x': 2174.615105346898, 'img1_y': 1994.8799720360635, 'img2_x': 1393.670078588304, 'img2_y': 2052.157827491442}, {'point_id': 2, 'img1_x': 2850.4937997203615, 'img1_y': 1296.0901354804487, 'img2_x': 1996.2331179788835, 'img2_y': 1412.93696060942}, {'point_id': 3, 'img1_x': 2804.6715153560585, 'img1_y': 1289.2167928258032, 'img2_x': 1957.284176269226, 'img2_y': 1406.063617954775}, {'point_id': 4, 'img1_x': 2639.7112916445694, 'img1_y': 1339.621305626536, 'img2_x': 1808.361752085243, 'img2_y': 1433.5569885733566}, {'point_id': 5, 'img1_x': 1940.9214550889544, 'img1_y': 1247.976736897931, 'img2_x': 1166.8497709850062, 'img2_y': 1314.4190492261696}, {'point_id': 6, 'img1_x': 1045.0957957668388, 'img1_y': 1181.5344245696924, 'img2_x': 197.7084566800063, 'img2_y': 1179.2433103514772}, {'point_id': 7, 'img1_x': 1216.9293621329734, 'img1_y': 1516.037100429101, 'img2_x': 385.5798225736471, 'img2_y': 1548.112699484113}, {'point_id': 8, 'img1_x': 1840.1124294874887, 'img1_y': 1552.6949279205433, 'img2_x': 1066.0407453835405, 'img2_y': 1605.3905549394913}, {'point_id': 9, 'img1_x': 1837.8213152692736, 'img1_y': 1926.1465454896093, 'img2_x': 1061.4585169471102, 'img2_y': 1983.4244009449876}, {'point_id': 10, 'img1_x': 1216.9293621329734, 'img1_y': 1933.0198881442548, 'img2_x': 387.8709367918623, 'img2_y': 1983.4244009449876}, {'point_id': 11, 'img1_x': 1617.8743503206213, 'img1_y': 3390.1685309290774, 'img2_x': 800.2714960705853, 'img2_y': 3477.2308712212525}, {
    #     'point_id': 12, 'img1_x': 1553.7231522105976, 'img1_y': 3268.7394773636756, 'img2_x': 736.1202979605623, 'img2_y': 3358.0929318740655}, {'point_id': 13, 'img1_x': 2887.1516272118033, 'img1_y': 2446.229473024444, 'img2_x': 2023.7264885974646, 'img2_y': 2464.558386770165}, {'point_id': 14, 'img1_x': 2302.9175015669452, 'img1_y': 2487.469528952316, 'img2_x': 1503.6435610626304, 'img2_y': 2524.1273564437583}, {'point_id': 15, 'img1_x': 928.2489706378673, 'img1_y': 2072.777855455378, 'img2_x': 51.0771467142381, 'img2_y': 2146.0935104382615}, {'point_id': 16, 'img1_x': 1613.292121884191, 'img1_y': 2521.836242225543, 'img2_x': 816.3092955980919, 'img2_y': 2599.7341256448576}, {'point_id': 17, 'img1_x': 1789.7079166867559, 'img1_y': 2812.807747938865, 'img2_x': 997.3073188370863, 'img2_y': 2879.2500602671034}, {'point_id': 18, 'img1_x': 1638.4943782845573, 'img1_y': 2764.6943493563467, 'img2_x': 839.2204377802427, 'img2_y': 2838.0100043392313}, {'point_id': 19, 'img1_x': 1375.0162431898175, 'img1_y': 2872.3767176124584, 'img2_x': 552.8311605033514, 'img2_y': 2980.059085868569}, {'point_id': 20, 'img1_x': 1035.9313388939781, 'img1_y': 2947.9834868135576, 'img2_x': 165.6328576249948, 'img2_y': 3085.450339906465}, {'point_id': 21, 'img1_x': 953.4512270382337, 'img1_y': 2870.0856033942428, 'img2_x': 67.11494624174384, 'img2_y': 2998.3879996142905}, {'point_id': 22, 'img1_x': 2131.0839352008106, 'img1_y': 833.2850634009924, 'img2_x': 1347.8477942240015, 'img2_y': 931.8029747842429}]

    # 3
    x1, x2 = intP2coords(interestPoints)

    # 4
    H2to1, inliers = computeH_ransac(x1, x2, opts)

    # 5
    image2_warped = cv2.warpPerspective(image2, H2to1, dsize=(
        image1.shape[1], image1.shape[0]))

    # 6
    image2_warped = rescaleImg(image2_warped, 15)
    image1 = rescaleImg(image1, 15)

    # 7
    zeroInd = np.where(image2_warped == 0)
    image2_warped[zeroInd] = image1[zeroInd]

    # 8
    cv2.imshow("image 2 warped", image2_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    panorama("livingroom_left.jpg", "livingroom_right.jpg")
