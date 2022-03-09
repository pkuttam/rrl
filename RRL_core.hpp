//
// Created by jhwangbo on 11.11.16.
//

#ifndef RAI_CORE_HPP
#define RAI_CORE_HPP
#include <sys/stat.h>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <raiCommon/utils/rai_timer/RAI_timer_ToInclude.hpp>
#include <raiCommon/utils/rai_logger/RAI_logger_ToInclude.hpp>
#include <raiCommon/utils/rai_graph/gnuplotter.hpp>
#include <glog/logging.h>
#include <omp.h>
#include <RRL_Tensor.hpp>
#include <boost/filesystem.hpp>
std::string RAI_LOG_PATH; /// logging directory
std::string RAI_ROOT_PATH; /// the main RAI directory

using rai::Utils::graph;
using rai::Utils::logger;
using rai::Utils::timer;

inline void rrl_init() {
    std::string rai_root="/home/grasping/project/legged/rrl/out/logs";
  if(!RAI_LOG_PATH.empty()) return;
  RAI_LOG_PATH = std::string("/logsNplots/" + timer->getCurrentDataAndTime());
  //LOG_IF(FATAL, std::string(rai_root).empty())
  //<< "Environment variable RAI_ROOT is not defined. Make sure it is written in ~/.bashrc file. If you are using clion, add it manually in the execution";
  RAI_LOG_PATH = std::string(rai_root) + RAI_LOG_PATH;
  RAI_ROOT_PATH = std::string(rai_root);
  //mkdir(RAI_LOG_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  boost::filesystem::create_directories(RAI_LOG_PATH.c_str());
  std::ostringstream glogPath;
  glogPath << RAI_LOG_PATH << "/";
  FLAGS_log_dir = glogPath.str();
  std::cout << "logging directory: " << glogPath.str() << std::endl;
  google::SetLogDestination(google::INFO, FLAGS_log_dir.c_str());
  google::InitGoogleLogging("debug");
  google::LogToStderr();
  google::SetCommandLineOption("GLOG_minloglevel", "0");
  graph->setLogPath(RAI_LOG_PATH);
  logger->setLogPath(RAI_LOG_PATH);
  timer->setLogPath(RAI_LOG_PATH);
  std::cout<<"hello\n";
}

#endif //RAI_RAI_UTILS_HPP
