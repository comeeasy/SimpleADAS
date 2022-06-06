import cv2
import numpy as np

if __name__ == "__main__":
    img = np.zeros((512, 512, 3))

    cv2.rectangle(img, (128, 128), (512-128, 512-128), (0, 255, 0), 3, -1)

    from_points = [
        [128, 128],
        [128, 512-128],
        [512-128, 128],
        [512-128, 512-128]
    ]
    to_points = [
        [50, 50],
        [50, 500],
        [400, 50],
        [400, 500]
    ]

    M = cv2.getPerspectiveTransform(
        np.array(from_points, dtype=np.float32),
        np.array(to_points, dtype=np.float32)
    )
    print(M)

    inv_M = np.linalg.inv(M)
    print(inv_M)

    bef_coor = np.array([50, 500, 1])
    aft_coor = np.matmul(inv_M, np.transpose(bef_coor))

    print(aft_coor)


    cv2.imshow("img", img)
    cv2.waitKey(0)