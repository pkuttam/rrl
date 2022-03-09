#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <cstdlib>

namespace rrl {
    namespace op {

        class compatibilty {
        public:
            int pkuIntRand(const int min, const int max){
                boost::uniform_int<> min2max(min,max);
                boost::variate_generator<boost::mt19937,boost::uniform_int<>> dice(rng,min2max);
                return dice();
            }
        private:
            boost::mt19937 rng(time(NULL));

        };
    }
}
