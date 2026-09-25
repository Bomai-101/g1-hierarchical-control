#include <iostream>
#include <functional>
#include <memory>
#include <mutex>
#include <cstdint>
#include <unistd.h>
#include <fstream>
#include <cmath>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;


// ============================================================
// Constants
// ============================================================

const int G1_NUM_MOTOR = 29;

const int LEFT_KNEE  = 3;
const int RIGHT_KNEE = 9;

const int Q_OFFSET    = 0;
const int DQ_OFFSET   = G1_NUM_MOTOR;       // 29
const int RPY_OFFSET  = G1_NUM_MOTOR * 2;   // 58
const int GYRO_OFFSET = RPY_OFFSET + 3;      // 61

const int OBS_SIZE =
    G1_NUM_MOTOR * 2 + 3 + 3;               // 64

const int ACTION_SIZE = G1_NUM_MOTOR;

const float KP = 20.0f;
const float KD = 2.0f;

static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string HG_CMD_TOPIC   = "rt/lowcmd";


// ============================================================
// Shared robot state
// ============================================================

float current_q[G1_NUM_MOTOR] = {};
float current_dq[G1_NUM_MOTOR] = {};

float initial_q[G1_NUM_MOTOR] = {};

// Latest policy observation
float observation[OBS_SIZE] = {};

bool state_received = false;

uint8_t current_mode_machine = 0;

std::mutex state_mutex;


// ============================================================
// CRC
// ============================================================

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len)
{
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;

    const uint32_t dwPolynomial = 0x04c11db7;

    for (uint32_t i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];

        for (uint32_t bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }

            xbit >>= 1;
        }
    }

    return CRC32;
}


// ============================================================
// Scripted Policy
// Observation[64] -> Action[29]
// ============================================================

void ScriptedPolicy(
    const float observation[OBS_SIZE],
    const float reference_q[G1_NUM_MOTOR],
    float action[ACTION_SIZE]
)
{
    // ========================================================
    // 1. Default action:
    // all joints stay at reference posture
    // ========================================================
    for (int i = 0; i < ACTION_SIZE; i++)
    {
        action[i] = 0.0f;
    }

    // ========================================================
    // 2. Read joint-level observation
    // ========================================================

    float left_knee_q =
        observation[LEFT_KNEE];

    float left_knee_dq =
        observation[DQ_OFFSET + LEFT_KNEE];

    float right_knee_q =
        observation[RIGHT_KNEE];

    float right_knee_dq =
        observation[DQ_OFFSET + RIGHT_KNEE];

    // ========================================================
    // 3. Read body-level observation
    // ========================================================

    float body_pitch =
        observation[RPY_OFFSET + 1];

    float pitch_rate =
        observation[GYRO_OFFSET + 1];

    // ========================================================
    // 4. Relative joint position
    // ========================================================

    float left_relative_q =
        left_knee_q -
        reference_q[LEFT_KNEE];

    float right_relative_q =
        right_knee_q -
        reference_q[RIGHT_KNEE];

    // ========================================================
    // 5. Detect joint motion
    // ========================================================

    bool joint_moving_fast =
        std::fabs(left_knee_dq) > 0.5f ||
        std::fabs(right_knee_dq) > 0.5f;

    // ========================================================
    // 6. Detect large body motion
    //
    // Teaching thresholds for MuJoCo only.
    // ========================================================

    bool body_motion_large =
        std::fabs(body_pitch) > 0.15f ||
        std::fabs(pitch_rate) > 0.8f;

    // ========================================================
    // 7. Whole-body-aware policy decision
    // ========================================================

    if (body_motion_large)
    {
        action[LEFT_KNEE]  = 0.03f;
        action[RIGHT_KNEE] = 0.03f;
    }
    else if (joint_moving_fast)
    {
        action[LEFT_KNEE]  = 0.05f;
        action[RIGHT_KNEE] = 0.05f;
    }
    else
    {
        action[LEFT_KNEE]  = 0.10f;
        action[RIGHT_KNEE] = 0.10f;
    }


    // Currently kept for inspection / future policy logic
    (void)left_relative_q;
    (void)right_relative_q;
}


// ============================================================
// LowState Callback
// ============================================================

void LowStateHandler(const void *message)
{
    LowState_ low_state =
        *(const LowState_ *)message;

    {
        std::lock_guard<std::mutex> lock(state_mutex);

        // ----------------------------------------------------
        // 1. Update current physical joint state
        // ----------------------------------------------------

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            current_q[i] =
                low_state.motor_state()[i].q();

            current_dq[i] =
                low_state.motor_state()[i].dq();
        }

        // ========================================================
        // IMU observation
        // ========================================================

        const auto &rpy =
            low_state.imu_state().rpy();

        const auto &gyro =
            low_state.imu_state().gyroscope();

        for (int i = 0; i < 3; i++)
        {
            observation[RPY_OFFSET + i] =
                rpy[i];

            observation[GYRO_OFFSET + i] =
                gyro[i];
        }

        // ----------------------------------------------------
        // 2. Keep current machine mode
        // ----------------------------------------------------

        current_mode_machine =
            low_state.mode_machine();


        // ----------------------------------------------------
        // 3. Build latest observation
        //
        // observation[0..28]  = q
        // observation[29..57] = dq
        // ----------------------------------------------------

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            observation[i] =
                current_q[i];

            observation[G1_NUM_MOTOR + i] =
                current_dq[i];
        }


        // ----------------------------------------------------
        // 4. First state only:
        // save fixed reference posture
        // ----------------------------------------------------

        if (!state_received)
        {
            for (int i = 0; i < G1_NUM_MOTOR; i++)
            {
                initial_q[i] =
                    current_q[i];
            }

            state_received = true;

            std::cout
                << "\nInitial state received"
                << "\nInitial left knee q = "
                << initial_q[LEFT_KNEE]
                << std::endl;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        std::cout
            << "Usage: g1_day4_controller network_interface"
            << std::endl;

        return 0;
    }

    std::string networkInterface = argv[1];


    // ========================================================
    // 1. DDS initialization
    // ========================================================

    ChannelFactory::Instance()->Init(
        1,
        networkInterface
    );


    // ========================================================
    // 2. LowState subscriber
    // ========================================================

    auto lowstate_subscriber =
        std::make_shared<ChannelSubscriber<LowState_>>(
            HG_STATE_TOPIC
        );

    lowstate_subscriber->InitChannel(
        std::bind(
            &LowStateHandler,
            std::placeholders::_1
        ),
        1
    );


    // ========================================================
    // 3. LowCmd publisher
    // ========================================================

    auto lowcmd_publisher =
        std::make_shared<ChannelPublisher<LowCmd_>>(
            HG_CMD_TOPIC
        );

    lowcmd_publisher->InitChannel();


    // ========================================================
    // 4. Wait for first state
    // ========================================================

    std::cout
        << "Waiting for LowState..."
        << std::endl;

    while (true)
    {
        bool ready;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ready = state_received;
        }

        if (ready)
        {
            break;
        }

        usleep(1000);
    }


    // ========================================================
    // 5. Save fixed reference posture
    // ========================================================

    float reference_q[G1_NUM_MOTOR];

    {
        std::lock_guard<std::mutex> lock(state_mutex);

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            reference_q[i] =
                initial_q[i];
        }
    }


    // ========================================================
    // 6. CSV logging
    // ========================================================

    std::ofstream log_file(
        "day4_scripted_policy.csv"
    );

    if (!log_file.is_open())
    {
        std::cerr
            << "Failed to open CSV file"
            << std::endl;

        return 1;
    }

    log_file
        << "time,target_q,current_q,error,dq\n";


    // ========================================================
    // 7. Experiment configuration
    // ========================================================

    const double dt = 0.002;
    const double duration = 5.0;

    double time = 0.0;

    std::cout
        << "Starting Day 4 scripted policy..."
        << std::endl;


    // ========================================================
    // 8. Control loop
    // ========================================================

    while (time < duration)
    {
        // ----------------------------------------------------
        // A. Take a snapshot of latest observation
        // ----------------------------------------------------

        float obs_snapshot[OBS_SIZE];

        uint8_t mode_machine;

        {
            std::lock_guard<std::mutex> lock(state_mutex);

            for (int i = 0; i < OBS_SIZE; i++)
            {
                obs_snapshot[i] =
                    observation[i];
            }

            mode_machine =
                current_mode_machine;
        }


        // ----------------------------------------------------
        // B. Policy:
        // Observation -> Action
        // ----------------------------------------------------

        float action[ACTION_SIZE] = {};

        ScriptedPolicy(
            obs_snapshot,
            reference_q,
            action
        );


        // ----------------------------------------------------
        // C. Convert action into joint targets
        //
        // q_target =
        // reference posture + policy action
        // ----------------------------------------------------

        float q_target[G1_NUM_MOTOR];

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            q_target[i] =
                reference_q[i] + action[i];
        }


        // ----------------------------------------------------
        // D. Read knee values from observation
        // for logging
        // ----------------------------------------------------

        float knee_q =
            obs_snapshot[LEFT_KNEE];

        float knee_dq =
            obs_snapshot[DQ_OFFSET + LEFT_KNEE];

        float knee_target =
            q_target[LEFT_KNEE];

        float error =
            knee_target - knee_q;


        // ----------------------------------------------------
        // E. Build LowCmd
        // ----------------------------------------------------

        LowCmd_ command{};

        command.mode_pr() = 0;

        command.mode_machine() =
            mode_machine;


        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            command.motor_cmd().at(i).mode() =
                1;

            command.motor_cmd().at(i).q() =
                q_target[i];

            command.motor_cmd().at(i).dq() =
                0.0f;

            command.motor_cmd().at(i).kp() =
                KP;

            command.motor_cmd().at(i).kd() =
                KD;

            command.motor_cmd().at(i).tau() =
                0.0f;
        }


        // ----------------------------------------------------
        // F. CRC
        // ----------------------------------------------------

        command.crc() =
            Crc32Core(
                reinterpret_cast<uint32_t *>(&command),
                (sizeof(command) >> 2) - 1
            );


        // ----------------------------------------------------
        // G. Send command
        // ----------------------------------------------------

        lowcmd_publisher->Write(command);


        // ----------------------------------------------------
        // H. Log knee behaviour
        // ----------------------------------------------------

        log_file
            << time << ","
            << knee_target << ","
            << knee_q << ","
            << error << ","
            << knee_dq
            << "\n";


        time += dt;

        usleep(2000);
    }


    // ========================================================
    // 9. Finish
    // ========================================================

    log_file.close();

    std::cout
        << "\nDay 4 experiment finished."
        << std::endl;

    std::cout
        << "Saved: day4_scripted_policy.csv"
        << std::endl;

    return 0;
}