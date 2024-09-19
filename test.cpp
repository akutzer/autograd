#include <memory>
#include <iostream>
#include <vector>


int main(int argc, char const *argv[])
{

    std::weak_ptr<int> wptr;
    std::shared_ptr<int> sptr;
    {
        auto ptr = std::make_shared<int>(1);
        
        wptr = ptr;
        sptr = wptr.lock();
        auto ptr3 = ptr;
        std::shared_ptr<int> spt = wptr.lock();
    }
    // auto ptr2 = std::make_wptr;

    // std::vector<std::shared_ptr<int>> vec;

    // vec.push_back(std::move(ptr2));
    // vec.pop_back();

    // ptr2.reset();
    // vec.at(0).reset();

    std::cout << wptr.use_count() << wptr.expired() << std::endl;

    return 0;
}
