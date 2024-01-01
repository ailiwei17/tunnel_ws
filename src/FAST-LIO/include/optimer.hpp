#include<deque>

namespace optimer
{
class MeanFilter{
private:
    std::deque<double> m_dq;
    int m_size;
    double sum;
public:
    MeanFilter():sum(0.0){};
    void init(int size){
        m_size = size;
    }
    void push(double value){
        m_dq.push_back(value);
        sum += value;
        if(m_dq.size() > m_size){
            sum-=m_dq.front();
            m_dq.pop_front();
        }
    }
    double get(){
        return sum/m_dq.size();
    }
};
    
} // namespace name

