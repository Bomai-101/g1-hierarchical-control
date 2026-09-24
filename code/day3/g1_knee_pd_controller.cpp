#include <iostream>
#include <functional>
#include <memory>
#include <mutex>
#include <cstdint>
#include <unistd.h>
#include <fstream>

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
const int LEFT_KNEE = 3;

const float TARGET_OFFSET = 0.1f;

const float HOLD_KP = 20.0f;
const float HOLD_KD = 2.0f;

const float KNEE_KP = 20.0f;
const float KNEE_KD = 2.0f;

static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string HG_CMD_TOPIC   = "rt/lowcmd";


// ============================================================
// Shared State
// ============================================================

float current_q[G1_NUM_MOTOR] = {};
float current_dq[G1_NUM_MOTOR] = {};

float initial_q[G1_NUM_MOTOR] = {};

float q_target = 0.0f;

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
// LowState Callback
// ============================================================

void LowStateHandler(const void *message)
{
    LowState_ low_state = *(const LowState_ *)message;

    float log_q;
    float log_dq;
    float log_target;

    {
        std::lock_guard<std::mutex> lock(state_mutex);

        // Update all 29 joint states
        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            current_q[i] =
                low_state.motor_state()[i].q();

            current_dq[i] =
                low_state.motor_state()[i].dq();
        }

        // Keep current machine mode
        current_mode_machine =
            low_state.mode_machine();


        // First LowState only:
        // save initial posture and create fixed knee target
        if (!state_received)
        {
            for (int i = 0; i < G1_NUM_MOTOR; i++)
            {
                initial_q[i] = current_q[i];
            }

            q_target =
                initial_q[LEFT_KNEE] + TARGET_OFFSET;

            state_received = true;

            std::cout
                << "\nInitial state received"
                << "\nInitial knee q = "
                << initial_q[LEFT_KNEE]
                << "\nKnee target q = "
                << q_target
                << "\n"
                << std::endl;
        }


        // Copy values for logging
        log_q = current_q[LEFT_KNEE];
        log_dq = current_dq[LEFT_KNEE];
        log_target = q_target;
    }


    // Knee tracking error
    float error =
        log_target - log_q;


    // Reduce console logging frequency
    static int counter = 0;

    if (++counter >= 500)
    {
        counter = 0;

        std::cout
            << "Target q = " << log_target
            << ", Current q = " << log_q
            << ", Error = " << error
            << ", dq = " << log_dq
            << std::endl;
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
            << "Usage: g1_knee_state_reader network_interface"
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
    // 4. Wait for first valid LowState
    // ========================================================

    std::cout << "Waiting for LowState..." << std::endl;

    while (true)
    {
        bool ready = false;

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
    // 5. Save fixed initial posture
    // ========================================================

    float hold_q[G1_NUM_MOTOR];
    float knee_target;

    {
        std::lock_guard<std::mutex> lock(state_mutex);

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            hold_q[i] = initial_q[i];
        }

        knee_target = q_target;
    }


    // ========================================================
    // 6. CSV logging
    // ========================================================

    std::ofstream log_file("knee_tracking.csv");

    if (!log_file.is_open())
    {
        std::cerr
            << "Failed to open knee_tracking.csv"
            << std::endl;

        return 1;
    }

    log_file
        << "time,target_q,current_q,error,dq\n";


    // ========================================================
    // 7. Experiment configuration
    // ========================================================

    const double dt = 0.002;       // 2 ms
    const double duration = 5.0;   // 5 seconds

    double time = 0.0;


    std::cout
        << "Starting PD tracking..."
        << std::endl;

    std::cout
        << "Kp = " << KNEE_KP
        << ", Kd = " << KNEE_KD
        << ", Target = " << knee_target
        << std::endl;


    // ========================================================
    // 8. 500 Hz control loop
    // ========================================================

    while (time < duration)
    {
        float knee_q;
        float knee_dq;

        uint8_t mode_machine;


        // ----------------------------------------------------
        // Read latest feedback state
        // ----------------------------------------------------

        {
            std::lock_guard<std::mutex> lock(state_mutex);

            knee_q =
                current_q[LEFT_KNEE];

            knee_dq =
                current_dq[LEFT_KNEE];

            mode_machine =
                current_mode_machine;
        }


        float error =
            knee_target - knee_q;


        // ----------------------------------------------------
        // Build LowCmd
        // ----------------------------------------------------

        LowCmd_ command{};

        command.mode_pr() = 0;

        command.mode_machine() =
            mode_machine;


        // ----------------------------------------------------
        // All joints hold their initial positions
        // ----------------------------------------------------

        for (int i = 0; i < G1_NUM_MOTOR; i++)
        {
            command.motor_cmd().at(i).mode() =
                1;

            command.motor_cmd().at(i).q() =
                hold_q[i];

            command.motor_cmd().at(i).dq() =
                0.0f;

            command.motor_cmd().at(i).kp() =
                HOLD_KP;

            command.motor_cmd().at(i).kd() =
                HOLD_KD;

            command.motor_cmd().at(i).tau() =
                0.0f;
        }


        // ----------------------------------------------------
        // Left knee experiment
        // ----------------------------------------------------

        command.motor_cmd().at(LEFT_KNEE).q() =
            knee_target;

        command.motor_cmd().at(LEFT_KNEE).dq() =
            0.0f;

        command.motor_cmd().at(LEFT_KNEE).kp() =
            KNEE_KP;

        command.motor_cmd().at(LEFT_KNEE).kd() =
            KNEE_KD;

        command.motor_cmd().at(LEFT_KNEE).tau() =
            0.0f;


        // ----------------------------------------------------
        // CRC
        // ----------------------------------------------------

        command.crc() =
            Crc32Core(
                reinterpret_cast<uint32_t *>(&command),
                (sizeof(command) >> 2) - 1
            );


        // ----------------------------------------------------
        // Send LowCmd
        // ----------------------------------------------------

        lowcmd_publisher->Write(command);


        // ----------------------------------------------------
        // Save experiment data
        // ----------------------------------------------------

        log_file
            << time << ","
            << knee_target << ","
            << knee_q << ","
            << error << ","
            << knee_dq
            << "\n";


        // ----------------------------------------------------
        // Advance experiment time
        // ----------------------------------------------------

        time += dt;

        usleep(2000);
    }


    // ========================================================
    // 9. Finish experiment
    // ========================================================

    log_file.close();

    std::cout
        << "\nExperiment finished."
        << std::endl;

    std::cout
        << "Saved: knee_tracking_kp20_kd2.csv"
        << std::endl;


    return 0;
}