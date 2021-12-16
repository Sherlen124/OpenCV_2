#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


const int matrix_size = 10;
float sigma = 10;
float blur_matrix[matrix_size][matrix_size];

void Blured_Picture(Mat& orig_image) {
    
    Mat blur_result = orig_image;
    int matrix_x = 0;
    int matrix_y = 0;
    
    double summ = 0;

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            blur_matrix[i][j] = exp(-(pow(matrix_x, 2) + pow(matrix_y, 2)) / (2 * pow(sigma, 2))) / (2 * M_PI * pow(sigma, 2));
            summ += blur_matrix[i][j];
            matrix_x++;
        }
        matrix_x = 0;
        matrix_y++;
    }
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            blur_matrix[i][j] /= summ;
        }
    }
    // bluring image
    for (int i = 0; i < orig_image.rows; i++) {
        for (int j = 0; j < orig_image.cols; j++) {
            
            matrix_x = j - matrix_size;
            matrix_y = i + matrix_size;
            
            double coeff_R = 0;
            double coeff_G = 0;
            double coeff_B = 0;

            for (int ii = 0; ii < matrix_size; ii++) {
                for (int jj = 0; jj < matrix_size; jj++) {
                    int pointy_ = matrix_y;
                    if (matrix_y >= orig_image.rows) {
                        matrix_y = orig_image.rows - (matrix_y - (orig_image.rows - 1));
                    }
                    
                    coeff_B += blur_matrix[ii][jj] * orig_image.at<Vec3b>(abs(matrix_y), abs(matrix_x))[0];
                    coeff_G += blur_matrix[ii][jj] * orig_image.at<Vec3b>(abs(matrix_y), abs(matrix_x))[1];
                    coeff_R += blur_matrix[ii][jj] * orig_image.at<Vec3b>(abs(matrix_y), abs(matrix_x))[2];
                    if (matrix_x >= orig_image.cols - 1)matrix_x = matrix_x--;
                    else matrix_x++;
                    matrix_y = pointy_;
                }
                matrix_y--;
                matrix_x = j - matrix_size;
            }
            blur_result.at<Vec3b>(i, j)[0] = coeff_B;
            blur_result.at<Vec3b>(i, j)[1] = coeff_G;
            blur_result.at<Vec3b>(i, j)[2] = coeff_R;
        }
    }
    
    namedWindow("Blur_Picture");
    imshow("Blur_Picture", blur_result);
    imwrite("Blur_Picture.jpg", blur_result);
    waitKey(0);

    return;
}

//Gradient function
void Gradient_Sobel(Mat& orig_image) {
    Mat grad_x = orig_image;
    Mat grad_y = orig_image;
    Mat grad_Summ = orig_image;
    
    //вычисляем производные по горизонтальному и вертикальному направлению
    for (int i = 0; i < orig_image.rows; i++) {
        for (int j = 0; j < orig_image.cols - 1; j++) {
            
            grad_x.at<uchar>(i, j) = (orig_image.at<uchar>(i, j + 1) - orig_image.at<uchar>(i, j));
        }
    }
    
    for (int i = 0; i < orig_image.rows - 1; i++) {
        for (int j = 0; j < orig_image.cols; j++) {
            grad_y.at<uchar>(i, j) = (orig_image.at<uchar>(i + 1, j) - orig_image.at<uchar>(i, j));

        }
    }
  
    //смешиваем х и у
    for (int i = 0; i < orig_image.rows - 1; i++) {
        for (int j = 0; j < orig_image.cols; j++) {
            grad_Summ.at<uchar>(i, j) = (grad_x.at<uchar>(i, j) + grad_y.at<uchar>(i, j));
        }
    }
    namedWindow("Gradient_Sobel");
    imshow("Gradient_Sobel", grad_Summ);
    imwrite("Gradient_Sobel.jpg", grad_Summ);
    waitKey(0);
    return;
}

int main()
{
    Mat original_image = imread("image.jpg", IMREAD_COLOR);            
    if (original_image.empty())                        
    {
        cout << "Could not open or find the image" << endl;
        cin.get();                                     
        return -1;
    }
    namedWindow("Original_Picture");                   
    imshow("Original_Picture", original_image);        
    waitKey(0);

    Blured_Picture(original_image);

    Mat gray;
    cvtColor(original_image, gray, COLOR_RGB2GRAY);
    Gradient_Sobel(gray);


    return 0;
}