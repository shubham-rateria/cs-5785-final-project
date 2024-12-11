#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

// https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html
// g++ -std=c++20  `pkg-config --cflags --libs opencv4` sim_ssim_all.cpp

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Function to compute MSSIM between two images
Scalar getMSSIM(const Mat &i1, const Mat &i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);

    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    Mat ssim_map;
    divide(t3, t1, ssim_map);

    Scalar mssim = mean(ssim_map);
    return mssim;
}

// Function to read all image files from a specified directory
vector<string> get_image_files(const string &folder)
{
    vector<string> image_paths;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (entry.is_regular_file())
        {
            string path = entry.path().string();
            if (path.size() >= 4 &&
                (path.compare(path.size() - 4, 4, ".jpg") == 0 ||
                 path.compare(path.size() - 5, 5, ".jpeg") == 0 ||
                 path.compare(path.size() - 4, 4, ".png") == 0 ||
                 path.compare(path.size() - 4, 4, ".bmp") == 0))
            {
                image_paths.push_back(path);
            }
        }
    }
    return image_paths;
}

// Function to load images
vector<Mat> load_images(const vector<string> &image_paths)
{
    vector<Mat> images;
    cout << "Loading images..." << endl;
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        Mat img = imread(image_paths[i], IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cerr << "Error: Could not load image " << image_paths[i] << endl;
        }
        else
        {
            images.push_back(img);
        }
        cout << "Loaded " << i + 1 << "/" << image_paths.size() << " images\r" << flush;
    }
    cout << endl
         << "All images loaded successfully." << endl;
    return images;
}

// Function to calculate similarity matrix and save to a file
void calculate_and_save_similarity_matrix(const vector<Mat> &images, const string &output_file)
{
    int num_images = images.size();
    vector<vector<double>> similarity_matrix(num_images, vector<double>(num_images, 0.0));

    cout << "Calculating similarity matrix..." << endl;

    // Calculate the similarity matrix
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            if (i == j)
            {
                similarity_matrix[i][j] = 1.0;
            }
            else
            {
                Scalar mssim = getMSSIM(images[i], images[j]);
                double score = mssim[0]; // Assuming a single channel
                similarity_matrix[i][j] = score;
                similarity_matrix[j][i] = score;
            }
        }
        // Print progress
        cout << "Processed " << i + 1 << "/" << num_images << " rows completed ("
             << fixed << setprecision(2)
             << (100.0 * (i + 1) / num_images) << "%)" << "\r" << flush;
    }
    cout << endl
         << "Similarity matrix calculation complete." << endl;

    // Save the matrix to a file
    ofstream ofs(output_file);
    if (!ofs)
    {
        cerr << "Error: Could not open file " << output_file << " for writing" << endl;
        return;
    }

    for (const auto &row : similarity_matrix)
    {
        for (double value : row)
        {
            ofs << value << ",";
        }
        ofs << "\n";
    }

    ofs.close();
    cout << "Similarity matrix saved to " << output_file << endl;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <image_folder> <output_file>" << endl;
        return -1;
    }

    string folder = argv[1];
    string output_file = argv[2];

    // Get list of image files
    cout << "Scanning directory for images..." << endl;
    vector<string> image_paths = get_image_files(folder);
    if (image_paths.empty())
    {
        cerr << "Error: No images found in the folder " << folder << endl;
        return -1;
    }
    cout << "Found " << image_paths.size() << " images." << endl;

    // Load images
    vector<Mat> images = load_images(image_paths);
    if (images.empty())
    {
        cerr << "Error: Could not load any images." << endl;
        return -1;
    }

    // Calculate and save the similarity matrix
    calculate_and_save_similarity_matrix(images, output_file);

    return 0;
}
