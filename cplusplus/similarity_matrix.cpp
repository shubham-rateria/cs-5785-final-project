#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

mutex mtx; // Mutex for thread-safe matrix updates

// Function to read an image in grayscale
Mat read_image(const string &path)
{
    return imread(path, IMREAD_GRAYSCALE);
}

// Function to calculate the structural similarity between two images
double calculate_similarity(const Mat &img1, const Mat &img2)
{
    Mat diff;
    absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff / 255.0; // Normalize pixel values to [0, 1]
    Scalar s = sum(diff);
    double score = s[0] / (img1.rows * img1.cols);
    return score;
}

// Function to calculate a row of the similarity matrix
void calculate_similarity_row(int i, const vector<Mat> &images, vector<vector<double>> &similarity_matrix)
{
    int num_images = images.size();
    for (int j = i; j < num_images; j++)
    {
        if (i == j)
        {
            similarity_matrix[i][j] = 1.0; // Similarity with itself is 1
        }
        else
        {
            double score = calculate_similarity(images[i], images[j]);

            // Thread-safe matrix update
            lock_guard<mutex> lock(mtx);
            similarity_matrix[i][j] = score;
            similarity_matrix[j][i] = score;
        }
    }
}

int main()
{
    // Folder containing images
    string image_folder = "../data/college/Images/";
    vector<string> image_files = {"0001.jpg", "0003.jpg", "0002.jpg"}; // List of your image files

    // Load images
    vector<Mat> images;
    for (const auto &filename : image_files)
    {
        string path = image_folder + filename;
        Mat img = read_image(path);
        if (img.empty())
        {
            cerr << "Error reading image: " << path << endl;
            return -1;
        }
        images.push_back(img);
    }

    int num_images = images.size();
    vector<vector<double>> similarity_matrix(num_images, vector<double>(num_images, 0.0));

    // Create and launch threads
    vector<thread> threads;
    for (int i = 0; i < num_images; i++)
    {
        threads.emplace_back(calculate_similarity_row, i, ref(images), ref(similarity_matrix));
    }

    // Join threads
    for (auto &th : threads)
    {
        th.join();
    }

    // Print the similarity matrix
    cout << "Similarity Matrix:" << endl;
    for (int i = 0; i < num_images; i++)
    {
        for (int j = 0; j < num_images; j++)
        {
            cout << similarity_matrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
