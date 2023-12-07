#include <vector>
#include "regress.hh"

using std::vector;

int main(int argc, char* argv[]) {
    
    constexpr float  kFLT_NAN   = std::numeric_limits<float>::quiet_NaN();

    // auto regressro_mod = Regress(3, 5);

    // vector<float> y{1.0, 20.0, 3.0, 4.0, 5.0};

    // vector<float> x1{1,2,3,4,5};
    // // vector<float> x1{kFLT_NAN, kFLT_NAN, kFLT_NAN, kFLT_NAN, kFLT_NAN};
    // vector<float> x2{0.1, 0.11, 0.31, 0.45, 0.55};

    // regressro_mod.set_const(0, 0.0f);
    // regressro_mod.set_x(1, x1.begin(), x1.end());
    // regressro_mod.set_x(2, x2.begin(), x2.end());

    // auto result = regressro_mod.solve(y.begin(), y.end());
    // auto b0 = result.beta.at(0);
    // auto b1 = result.beta.at(1);
    // auto b2 = result.beta.at(0);
    // auto res_nobs = result.nobs;
    // printf("debuginfo : b0 %f b1 %f b2 %f nobs %d\n", b0, b1, b2, res_nobs);

    auto regressro_mod = Regress(2, 3);
    vector<float> y{-0.906, 0.358, 0.359};
    vector<float> x0{-1, -0.737, 0.511};
    vector<float> x1{-0.0827, 0.0655, -0.562};

    regressro_mod.set_x(0, x0.begin(), x0.end());
    regressro_mod.set_x(1, x1.begin(), x1.end());
    auto result = regressro_mod.solve(y.begin(), y.end());
    auto b0 = result.beta.at(0);
    auto b1 = result.beta.at(1);
    auto res_nobs = result.nobs;
    printf("debuginfo : b0 %f b1 %f nobs %d\n", b0, b1,  res_nobs);

}