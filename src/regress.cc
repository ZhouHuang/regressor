#include "regress.hh"
#include <iostream>

Regress::Regress(int ncols, int nobs) : m_ncols{ncols}, m_nobs{nobs} {
    m_nobs_valid = 0;
    m_X.resize(m_nobs * m_ncols);
    m_y.resize(m_nobs);
    m_x_valid.resize(m_nobs * m_ncols, false);
    m_valid.resize(m_nobs, false);
    m_res.beta.resize(m_ncols, std::numeric_limits<float>::quiet_NaN());
    m_res.tstats.resize(m_ncols, std::numeric_limits<float>::quiet_NaN());
}

const RegResult& Regress::get() const { return m_res; }

void Regress::calc(StatsOption option) {  
    // 检查是否已经计算过解，如果是则直接返回  
    if (m_solved) {  
        return;  
    }  

    // filter nan
    for(int i = 0; i<m_nobs; ++i) {
        bool valid = std::isfinite(m_y.at(i));

        if (!valid) continue;

        for(int j = 0; j<m_ncols; ++j) {
            int shift_pos = i + m_nobs * j;
            valid = valid && std::isfinite(m_X.at(shift_pos));
            if (!valid) break;
        }
        if (!valid) continue;

        m_valid.at(i) = valid;
        for(int j = 0; j<m_ncols; ++j) {
            int shift_pos = i + m_nobs *j;
            m_x_valid.at(shift_pos) = valid;
        }
        ++m_nobs_valid;
    }

    m_res.nobs = m_nobs_valid;
    m_res.r2 = std::numeric_limits<float>::quiet_NaN();

    if (m_nobs_valid < 1) return;

    m_X_filtered.reserve(m_nobs_valid * m_ncols);
    m_y_filtered.reserve(m_nobs_valid);
    for(int i = 0; i<m_valid.size(); ++i) {
        auto valid = m_valid.at(i);
        if (!valid) continue;
        m_y_filtered.push_back(m_y.at(i));
    }

    for(int i = 0; i<m_x_valid.size(); ++i) {
        auto valid = m_x_valid.at(i);
        if (!valid) continue;
        m_X_filtered.push_back(m_X.at(i));
    }

    Eigen::Map<Eigen::MatrixXd> matX(m_X_filtered.data(), m_nobs_valid, m_ncols);  
    std::cout << "X : \n";
    std::cout << matX << '\n';
    
    Eigen::Map<Eigen::VectorXd> matY(m_y_filtered.data(), m_nobs_valid, 1);
    std::cout << "Y : \n";
    std::cout << matY << '\n';

    m_svd.compute(matX);

    auto V = m_svd.matrixV(); // 获取V矩阵（左奇异向量矩阵）  
    auto vec_S = m_svd.singularValues(); // 获取S奇异值向量
    auto U = m_svd.matrixU(); // 获取U矩阵（右奇异向量矩阵）  

    Eigen::MatrixXd S(vec_S.size(), vec_S.size());  
    // 将v1中的元素设置到主对角线上  
    S.triangularView<Eigen::Lower>().setZero();  
    S.triangularView<Eigen::Upper>().setZero();  
    S.diagonal().segment(0, vec_S.size()) = vec_S;  

    Eigen::VectorXd beta = (V * S.inverse()) * U.transpose() * matY; // 求解回归系数 

    std::cout << "beta : \n";
    std::cout << beta << '\n';

    if (kComputeR2 == option) {

    }

    // 将回归系数存入m_res对象的beta成员变量中  
    for(int i = 0; i<m_ncols; ++i) {
        m_res.beta[i] = beta[i];
    }
    if (kComputeT == option) {

    }
    
 }