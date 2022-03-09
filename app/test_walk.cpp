//
// Created by grasping on 04/03/22.
//


// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// commons
//#include "common/enumeration.hpp"
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
//#include <experimental/filesystem>


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

#define nThread 40
int main(int argc, char *argv[]) {
    rrl_init();

    ////////////////////////// Define task ////////////////////////////
    std::string urdfPath;
    std::string actuatorPath;

    urdfPath = "/home/grasping/project/legged/rrl/rsc/robot/c100/urdf/anymal_minimal.urdf";
    actuatorPath = "/home/grasping/project/legged/rrl/include/hwangbo/data/seaModel_10000.txt";
    //std::cout <<"world\n";
    raisim::World::setActivationKey("/home/grasping/license/activation.raisim");
    Task_ task(false, 1, urdfPath, actuatorPath);
    //std::cout <<"world1\n";
    task.setDiscountFactor(0.99);
//  task.setControlUpdate_dt(0.0025);
    task.setControlUpdate_dt(0.05);

    task.setTimeLimitPerEpisode(7.0);
    task.setRealTimeFactor(1.0);
    task.setNoiseFtr(1.0);

    Task_::GeneralizedCoordinate initialPos;
    Task_::GeneralizedVelocities initialVel;
    Task_::Vector12d jointTorque;

    initialPos << 0.0, 0.0, 0.14, 1.0, 0.0, 0.0, 0.0,
            0, M_PI / 2.0, -M_PI,
            0, M_PI / 2.0, -M_PI,
            0.0, -M_PI / 2.0, M_PI,
            0.0, -M_PI / 2.0, M_PI;

    initialPos << 0.0, 0.0, 0.14,
            0.5401, -0.8353, 0.0, 0.0121,
            0, 0.4, -0.8,
            0, 0.4, -0.8,
            0.0, -0.4, 0.8,
            0.0, -0.4, 0.8;

    initialPos << 0.0, 0.0, 0.25,
            0.5184, 0.7746, 0.3107, 0.1867,
            -0.8009, 0.5985, -0.4031,
            -0.8960, -1.2083, 0.2099,
            0.0377, 0.7822, -0.8328,
            -0.0671, -0.3675, 0.7284;

    initialVel << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    task.setInitialState(initialPos, initialVel);


    NoiseCov covariance = NoiseCov::Identity();
    covariance *= 1.0;
    NormNoise actionNoise(covariance);

    /// Tensors
    State state_e;
    Task_::Action action_e;
    rrl::Tensor<Dtype, 3> State3D({StateDim, 1, 1}, "state");
    rrl::Tensor<Dtype, 3> Action3D({ActionDim, 1, 1}, "action");
    rrl::Tensor<Dtype, 2> State2D({StateDim, 1}, "state");
    rrl::Tensor<Dtype, 2> Action2D({ActionDim, 1}, "action");

    rai::Utils::logger->addVariableToLog(2, "Ncontacts", "");
    rai::Utils::logger->addVariableToLog(4, "speed", "");
    rai::Utils::logger->addVariableToLog(13, "command", "");

    rai::Utils::Graph::FigProp2D figprop;


    task.init0();
    task.setActionLimit(M_PI);

    int max = 50000;

    Eigen::Matrix<Dtype, -1, 1> stateSaveBuffer;
    Eigen::Matrix<Dtype, -1, 1> actionSaveBuffer;

    stateSaveBuffer.resize(96 * max);
    actionSaveBuffer.resize(12 * max);

    task.getState(state_e);

    Eigen::Matrix<double, 19, 1> defaultConfig;
    defaultConfig
            << 0.0, 0.0, 0.44, 1.0, 0.0, 0.0, 0.0, -0.15, 0.4, -0.8, 0.15, 0.4, -0.8, -0.15, -0.4, 0.8, 0.15, -0.4, 0.8;
    Eigen::Matrix<double, 18, 1> defaultVel;
    defaultVel.setZero();

    Eigen::Matrix<Dtype, 12, 1> jointNominalConfig;

    jointNominalConfig << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;

    Eigen::Matrix<Dtype, 12, 1> flipJointConfig;
    Eigen::Matrix<Dtype, 12, 1> flipJointConfig2;
    Eigen::Matrix<Dtype, 12, 1> sitJointConfig;
    Eigen::Matrix<Dtype, 12, 1> dampedJointPos_;
    dampedJointPos_ << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;
    sitJointConfig <<
                   -0.15, M_PI / 3.0, -2.,
            0.15, M_PI / 3.0, -2.,
            -0.15, -M_PI / 3.0, 2.,
            0.15, -M_PI / 3.0, 2.;

    flipJointConfig <<
                    0.0, 0.2, -0.3,
            0.5, 1.0, -2.45,
            0.0, -0.2, 0.3,
            0.5, -1.0, 2.45;

    flipJointConfig2 <<
                     0.0, 0.2, -0.3,
            0.5, 0.4, -0.8,
            0.0, -0.2, 0.3,
            0.5, -0.4, 0.8;

    task.setInitialState(defaultConfig, defaultVel);
    task.init0();

    /**
    for (int iterationNumber = 0; iterationNumber < max; iterationNumber++) {

        rai::TerminationType type = rai::TerminationType::not_terminated;
        Dtype cost;

        //// forward
        State2D = state_e;

        stateSaveBuffer.segment(96 * iterationNumber, 96) = state_e;
        actionSaveBuffer.segment(12 * iterationNumber, 12) = action_e;

//    policy_.forward(State2D, Action2D);
//    action_e = Action2D.eMat();

        action_e.setZero();

        task.step(action_e, state_e, type, cost);


    }
**/


    return 0;
}