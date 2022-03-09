#include<iostream>
#include <tensorflow/core/public/session.h>
#include <Eigen/Core>
#include <eigen3/Eigen/Core>
#include <RRL_Tensor.hpp>
#include <RRL_core.hpp>
#include <algorithm/TRPO_gae.hpp>
#include <function/common/ParameterizedFunction.hpp>
#include <function/common/ValueFunction.hpp>

#include <function/tensorflow/common/TensorFlowNeuralNetwork.hpp>
#include "function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"

#include <memory/Trajectory.hpp>
#include <tasks/common/Task.hpp>
#include <noiseModel/Noise.hpp>
#include <noiseModel/NormalDistributionNoise.hpp>
#include <noiseModel/NoNoise.hpp>
#include <common/VectorHelper.hpp>
//#include <common/enumeration.hpp>
#include <algorithm/common/LearningData.hpp>
#include <function/common/Policy.hpp>
#include <function/common/StochasticPolicy.hpp>
#include <memory/ReplayMemorySARS.hpp>
#include <memory/ReplayMemoryS.hpp>
#include <experienceAcquisitor/AcquisitorCommonFunc.hpp>
#include <experienceAcquisitor/Acquisitor.hpp>
#include <experienceAcquisitor/TrajectoryAcquisitor.hpp>
#include <experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include <experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <algorithm/common/PerformanceTester.hpp>
#include <algorithm/TRPO_gae.hpp>
#include <challenging/SimpleMLPLayer.hpp>

//#include "anymal_minimal.hpp"
//#include "anymal_challenging.hpp"
#include "hwangbo/anymal.hpp"
int main(){
    std::cout<<"hello"<<std::endl;
    return 0;
}