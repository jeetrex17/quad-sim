#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "drone_sim/quad_dynamics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad_sim_cpp, m) {
    m.doc() = "Quadrotor physics simulator (C++ core via pybind11)";

    py::class_<QuadState>(m, "QuadState")
        .def(py::init<>())
        .def_readwrite("px", &QuadState::px)
        .def_readwrite("py", &QuadState::py)
        .def_readwrite("pz", &QuadState::pz)
        .def_readwrite("vx", &QuadState::vx)
        .def_readwrite("vy", &QuadState::vy)
        .def_readwrite("vz", &QuadState::vz)
        .def_readwrite("qw", &QuadState::qw)
        .def_readwrite("qx", &QuadState::qx)
        .def_readwrite("qy", &QuadState::qy)
        .def_readwrite("qz", &QuadState::qz)
        .def_readwrite("wx", &QuadState::wx)
        .def_readwrite("wy", &QuadState::wy)
        .def_readwrite("wz", &QuadState::wz)
        .def("__repr__", [](const QuadState &s) {
            return "<QuadState pos=(" +
                std::to_string(s.px) + "," +
                std::to_string(s.py) + "," +
                std::to_string(s.pz) + ") z_vel=" +
                std::to_string(s.vz) + ">";
        });

    // Step function: state, motor_speeds[4], dt -> new state
    m.def("step", [](const QuadState &s,
                     const std::array<double,4> &motors,
                     double dt) {
        return qd_rk4(s, motors, dt);
    }, "Run one RK4 physics step",
       py::arg("state"), py::arg("motors"), py::arg("dt") = 0.005);

    // Expose physical constants so Python env uses same values as C++
    m.attr("MASS")    = QD_MASS;
    m.attr("GRAVITY") = QD_GRAVITY;
    m.attr("KT")      = QD_KT;
    m.attr("KQ")      = QD_KQ;
    m.attr("L")       = QD_L;
    m.attr("W_HOVER") = QD_W_HOVER;
}
