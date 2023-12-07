#include <vector>
#include <Eigen/Dense>


struct RegResult
{
    std::vector<double> beta;
    std::vector<double> tstats;
    double r2;
    int nobs;
};

class Regress
{
public:
    enum{
        kComputeT = 0x01,
        kComputeR2 = 0x02
    };
    using StatsOption = int;
    Regress(int ncols, int nobs);
    const RegResult& get() const;
    template<typename Iter> 
    const RegResult& solve(Iter ybeg, Iter yend, StatsOption option = 0)
    {
        m_solved = false;
        if (m_nobs != std::distance(ybeg, yend)){
            throw std::runtime_error("Regress::solve error. invalid input size.");
        }
        std::copy(ybeg, yend, m_y.begin());
        calc(option);
        return get();
    }
    template<typename Iter>
    Regress& set_x(int col, Iter beg, Iter end)
    {
        m_solved = false;
        m_x_chg = true;
        if (m_nobs != std::distance(beg, end)){
            throw std::runtime_error("Regress::set_x error. invalid input size.");
        }
        std::copy(beg, end, &m_X[col*m_nobs]);
        return *this;
    }
    Regress& set_const(int col, double value=1.0)
    {
        m_solved = false;
        m_x_chg = true;
        std::fill_n(&m_X[m_nobs * col], m_nobs, value);
        return *this;
    }
    bool get_svd_reset() const {return m_svd_reset;}
private:
    void calc(StatsOption);
    int m_ncols;
    int m_nobs;
    int m_nobs_valid{};
    std::vector<double> m_X{};
    std::vector<double> m_y{};
    std::vector<bool> m_x_valid{};
    std::vector<bool> m_valid{};
    std::vector<double> m_X_filtered{};
    std::vector<double> m_y_filtered{};
    RegResult m_res{};
    bool m_x_chg{false};
    bool m_solved{false};
    bool m_svd_reset{false};
    Eigen::BDCSVD<Eigen::MatrixXd> m_svd;
};