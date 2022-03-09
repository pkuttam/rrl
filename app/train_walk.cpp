//
// Created by grasping on 09/03/22.
//

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// commons
#include "raiCommon/enumeration.hpp"
#include <RRL_core.hpp>
// task
#include "hwangbo/anymal.hpp"
// noise model
#include "noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "function/tensorflow/ValueFunction_TensorFlow.hpp"

// algorithm
#include <experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include <algorithm/TRPO_gae.hpp>

// filesystem
#include <experimental/filesystem>

/// learning states
using Dtype = float;

/// shortcuts
using rrl::Task::ActionDim;
using rrl::Task::StateDim;
using rrl::Task::CommandDim;
using Task_ = rrl::Task::anymal<Dtype>;

using State = Task_::State;
using Action = Task_::Action;
using Command =  Task_::Command;
using VectorXD = Task_::VectorXD;
using MatrixXD = Task_::MatrixXD;
typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> NoiseCovariance;
using Policy_TensorFlow = rrl::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction_TensorFlow = rrl::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using ReplayMemorySARS = rrl::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rrl::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Noise = rrl::Noise::Noise<Dtype, ActionDim>;
using NormNoise = rrl::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = rrl::Noise::NormalDistributionNoise<Dtype, ActionDim>::CovarianceMatrix;

#define nThread 100

int main(int argc, char *argv[]) {
    rrl_init();
    omp_set_dynamic(0);
    omp_set_num_threads(nThread);

    RAIWARN(omp_get_max_threads())

    std::string urdfPath;
    std::string actuatorPath;
    std::string controllerPath;

    raisim::World::setActivationKey("/home/grasping/license/activation.raisim");

    urdfPath = "/home/grasping/project/legged/rrl/rsc/robot/c100/urdf/anymal_minimal.urdf";
    actuatorPath = "/home/grasping/project/legged/rrl/include/hwangbo/data/seaModel_10000.txt";

    std::vector<std::unique_ptr<Task_>> taskVec;

    std::unique_ptr<Task_> temp(new Task_(false,0,urdfPath,actuatorPath));
    taskVec.emplace_back((std::move(temp)));

    for (int i=1; i<nThread; i++){
        std::unique_ptr<Task_> temp(new Task_(false,i%10,urdfPath,actuatorPath));
        taskVec.emplace_back(std::move(temp));
    }

    std::vector<rrl::Task::Task<Dtype,StateDim,ActionDim,0> *> taskVector;

    for (auto &task : taskVec) {
        task->setDiscountFactor(0.998);
        task->setValueAtTerminalState(3.0);
        task->setControlUpdate_dt(0.005);
        task->setTimeLimitPerEpisode(4.0);
        task->setRealTimeFactor(1.0);
        task->setNoiseFtr(0.4);
        taskVector.push_back(task.get());
    }

    /// the first one is not noisified
    for (int i=1; i<taskVec.size(); i++)
        taskVec[i]->noisifyDynamics();

    //////////////////////////// Define Noise /////////////////////////////
    NoiseCov covariance = NoiseCov::Identity();
    std::vector<NormNoise> noiseVec(nThread, NormNoise(covariance));
    std::vector<NormNoise *> noiseVector;
    for (auto &noise : noiseVec)
        noiseVector.push_back(&noise);


    ////////////////////////// Define Function approximations //////////
    Vfunction_TensorFlow vfunction("gpu,0", "MLP", "tanh 1e-3 97 192 128 1", 0.001);
    Policy_TensorFlow policy("gpu,0", "MLP", "tanh 1e-3 97 256 128 12", 0.001);

    ////////////////////////// Acquisitor
    Acquisitor_ acquisitor;

    ////////////////////////// Algorithm ////////////////////////////////
    rrl::Algorithm::TRPO_gae<Dtype, StateDim, ActionDim>
            algorithm(taskVector, &vfunction, &policy, noiseVector, &acquisitor, 0.99, 0, 0, 10, 1.0, 0.004, false);

    /////////////////////// Plotting properties ////////////////////////
    rai::Utils::Graph::FigProp2D figurePropertiesEVP;
    figurePropertiesEVP.title = "Number of Episodes vs Performance";
    figurePropertiesEVP.xlabel = "N. Episodes";
    figurePropertiesEVP.ylabel = "Performance";

    constexpr int loggingInterval = 100;
    constexpr int iterlimit = 30000;
    ////////////////////////// Learning /////////////////////////////////

    for (auto &task : taskVec) {
        task->setcostScale1(0.3);
        task->setcostScale2(0.1);
    }

    double actionLimit = M_PI;
    for (int iterationNumber = 0; iterationNumber < iterlimit + 1; iterationNumber++) {

        LOG(INFO) << "iter :" << iterationNumber;
        LOG(INFO) << "Learning Rate: " << vfunction.getLearningRate();
        LOG(INFO) << "actionLimit: " << actionLimit;

        LOG(INFO) << "costScale1:" << taskVec[0]->getcostScale1();
        LOG(INFO) << "costScale2:" << taskVec[0]->getcostScale2();

        for (auto &task : taskVec) {
            task->increaseCostScale1(0.998); // TODO: slower?
            task->increaseCostScale2(0.998);
        }

        int nEpisode = 100000 / (taskVec[0]->timeLimit() / taskVec[0]->dt());
        algorithm.runOneLoop(nEpisode);

        if (iterationNumber % loggingInterval == 0 || iterationNumber == iterlimit) {
            policy.dumpParam(
                    RAI_LOG_PATH + "/recovery_policy_" + std::to_string(iterationNumber) + ".txt");
            vfunction.dumpParam(
                    RAI_LOG_PATH + "/value_" + std::to_string(iterationNumber) + ".txt");
        }
    }


}