#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <glog/logging.h>
#include "frontend/FullSystem.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"
#include "DatasetReader.h"
#include <sys/time.h>
#include <signal.h>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <deque>
#include <iostream>

using namespace std;
using namespace ldso;

std::atomic<bool> quit(false);

void sigintHandler(int) {
    quit.store(true);
}

class WebcamCapture {
public:
    WebcamCapture(int width, int height, float fps, std::shared_ptr<Undistort> undistorter)
        : width(width), height(height), fps(fps), undistorter(undistorter) {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Cannot open webcam" << endl;
            exit(1);
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, fps);
    }

    ImageAndExposure* getImage() {
        cv::Mat frame, gray;
        if (!cap.read(frame)) {
            cerr << "Failed to capture frame" << endl;
            return nullptr;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::system_clock::now().time_since_epoch()).count();
        MinimalImageB minImg((int)gray.cols, (int)gray.rows, gray.data);
        ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1.0f, timestamp / 1e9);
        return undistImg;
    }

private:
    cv::VideoCapture cap;
    int width;
    int height;
    float fps;
    std::shared_ptr<Undistort> undistorter;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;

    string calib = "./examples/EUROC/live.txt";
    string vignetteFile = "";
    string gammaFile = "";
    string vocPath = "./vocab/orbvoc.dbow3";
    string output_file = "results.txt";

    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);

    std::shared_ptr<Undistort> undistorter = std::shared_ptr<Undistort>(
        Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile));

    int w_out, h_out;
    Eigen::Matrix3f K;
    K = undistorter->getK().cast<float>();
    w_out = undistorter->getSize()[0];
    h_out = undistorter->getSize()[1];
    setGlobalCalib(w_out, h_out, K);

    shared_ptr<FullSystem> fullSystem(new FullSystem(voc));
    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if (!disableAllDisplay) {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(w_out, h_out, false));
        fullSystem->setViewer(viewer);
    }

    WebcamCapture webcam(640, 480, 30.0f, undistorter);

    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();
    int numFramesProcessed = 0;

    // FPS and processing time calculation variables
    const int WINDOW_SIZE = 30;
    std::deque<double> processing_times;
    double total_processing_time = 0.0;

    signal(SIGINT, sigintHandler);

    std::thread runthread([&]() {
        int id = 0;
        while (!quit.load()) {
            while (setting_pause) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            ImageAndExposure* img = webcam.getImage();
            if (img == nullptr) continue;

            // Measure SLAM processing time
            auto slam_start = std::chrono::high_resolution_clock::now();
            
            fullSystem->addActiveFrame(img, id);
            
            auto slam_end = std::chrono::high_resolution_clock::now();
            double processing_time = std::chrono::duration<double, std::milli>(slam_end - slam_start).count();

            // Update processing times
            processing_times.push_back(processing_time);
            total_processing_time += processing_time;

            if (processing_times.size() > WINDOW_SIZE) {
                total_processing_time -= processing_times.front();
                processing_times.pop_front();
            }

            id++;
            numFramesProcessed++;

            // Print average processing time and effective FPS every WINDOW_SIZE frames
            if (numFramesProcessed % WINDOW_SIZE == 0) {
                double avg_processing_time = total_processing_time / WINDOW_SIZE;
                double effective_fps = 1000.0 / avg_processing_time;  // Convert ms to seconds

                cout << "Average processing time: " << fixed << setprecision(2) 
                     << avg_processing_time << " ms" << endl;
                cout << "Effective FPS: " << fixed << setprecision(2) 
                     << effective_fps << endl;
            }

            delete img;

            if (fullSystem->initFailed || setting_fullResetRequested) {
                if (id < 250 || setting_fullResetRequested) {
                    cout << "RESETTING!" << endl;
                    fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                    if (viewer) {
                        viewer->reset();
                        fullSystem->setViewer(viewer);
                    }
                    setting_fullResetRequested = false;
                }
            }

            if (fullSystem->isLost) {
                cout << "Lost!" << endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 fps
        }

        fullSystem->blockUntilMappingIsFinished();

        // Save results and print statistics
        fullSystem->printResult(output_file, true);
        fullSystem->printResult(output_file + ".noloop", false);

        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        double numSecondsProcessed = (tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1e6;
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        
        std::ofstream statsFile(output_file + "_stats.txt");
        statsFile << fixed << setprecision(1);
        statsFile << "======================" << endl;
        statsFile << numFramesProcessed << " Frames (" << numFramesProcessed / numSecondsProcessed << " fps)" << endl;
        statsFile << setprecision(2);
        statsFile << MilliSecondsTakenSingle / numFramesProcessed << "ms per frame (single core)" << endl;
        statsFile << MilliSecondsTakenMT / (float) numFramesProcessed << "ms per frame (multi core)" << endl;
        statsFile << setprecision(3);
        statsFile << 1000 / (MilliSecondsTakenSingle / numFramesProcessed) << "x (single core)" << endl;
        statsFile << 1000 / (MilliSecondsTakenMT / numFramesProcessed) << "x (multi core)" << endl;
        statsFile << "======================" << endl;
        statsFile.close();

        cout << "Results and statistics saved." << endl;
    });

    if (viewer)
        viewer->run();

    runthread.join();

    cout << "EXIT NOW!" << endl;
    return 0;
}