
import svdsift
from datetime import datetime
import time
# main programme
def main(file1, file2, color, dispMin, dispMax,sigma):
    start= time.time()
    # load two images
    imgLeft, imgRight = svdsift.svd_main(dispMin, dispMax,file1,file2, sigma)
    
    
    #imgLeft = imgLeft[200:, 100:]
    #imgLeft = cv2.resize(imgLeft, (200,300) ,interpolation = cv2.INTER_AREA)
    #imgRight = cv2.resize(imgRight, (200,300) ,interpolation = cv2.INTER_AREA)
    
    # Default parameters
    K = -1
    lambda_ = -1
    lambda1 = -1
    lambda2 = -1
    params = match.Parameters(is_L2=True,
                              denominator=1,
                              edgeThresh=8,
                              lambda1=lambda1,
                              lambda2=lambda2,
                              K=K,
                              maxIter=10,
                              sigma_d = 14,
                              sigma_s = 3.8,
                              lambda_ancc = 1/30,
                              V_max = 16,
                              width = 3,
                              theta = 0.7,
                              gamma = 1
                              )
    
    # create match instance
    m = match.Match(imgLeft, imgRight, color)
    m.SetDispRange(dispMin, dispMax)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    m.kolmogorov_zabih()
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    m.saveDisparity("./results/disparity_{}_{}.jpg".format(current_time,sigma))
    end = time.time()
    print(end-start)
    

if __name__ == '__main__':
    
    filenames = ['./images/perra_9.png',"./images/perra_10.png"]
    is_color = True
    disMin = -16
    disMax = 16
    sigma = 200
    for i in range(0,len(filenames)):
        filename_left = filenames[i][0]
        filename_right = filenames[i][1]
    main(filename_left, filename_right, is_color, disMin, disMax,sigma)
