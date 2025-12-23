     return v;

}

class AGIMathematics {

private:

     std::vector<double> tempSubset;

public:

     double entropy(const std::vector<double> &data) {

         if (data.empty()) return 0.0;

         double sum = 0.0;

         for (auto d : data) sum += std::abs(d);

         if (sum <= 0.0) return 0.0;

         double result = 0.0;

         for (auto d : data) {

             double p = std::abs(d) / sum;

             if (p > 0.0) result -= p * std::log(p);

         }

         return result;

     }

     double integrated_information(const std::vector<double> &vec) {

         if (vec.empty()) return 0.0;

         size_t n = vec.size();

         size_t parts = std::max<size_t>(1, n / 2);

         double sys_ent = entropy(vec);

         double part_ent = 0.0;

         tempSubset.clear();

         tempSubset.reserve(n);

         for (size_t i = 0; i < parts; ++i) {

             tempSubset.clear();

             for (size_t j = i; j < vec.size(); j += parts) tempSubset.push_back(vec[j]);

             part_ent += entropy(tempSubset);

         }

         part_ent /= static_cast<double>(parts);

         return std::max(0.0, sys_ent - part_ent);

     }

};

class KnowledgeDNA {

public:

     int generation = 0;

