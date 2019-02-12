#define NDEBUG
#include <algorithm>
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/gil.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define FOR_ELSE(condition, for_block, else_block) \
    ([&]() { for                                   \
            condition for_block else_block         \
    }())

int main(int argc, char *argv[]) {
    boost::program_options::options_description options_description("option");
    options_description.add_options()("input_directory", boost::program_options::value<std::string>(), "directory of input data")(
        "output_directory", boost::program_options::value<std::string>(), "directory of output data")(
        "image_width", boost::program_options::value<int>()->default_value(256), "width of image that will be generated")(
        "image_height", boost::program_options::value<int>()->default_value(256), "height of image that will be generated")(
        "seq_lens", boost::program_options::value<std::vector<int>>()->multitoken()->default_value({5, 10}, "5, 10"), "sequence lengths of texts")(
        "num_data", boost::program_options::value<int>()->default_value(900000), "number of data that will be generated")(
        "num_retries", boost::program_options::value<int>()->default_value(100), "number of retries for locating bounding box");

    boost::program_options::variables_map variables_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options_description), variables_map);
    boost::program_options::notify(variables_map);

    std::vector<boost::filesystem::path> filenames;
    boost::copy(boost::make_iterator_range(boost::filesystem::recursive_directory_iterator(variables_map["input_directory"].as<std::string>()), {}) |
                    boost::adaptors::transformed([](const auto &entry) { return entry.path(); }) |
                    boost::adaptors::filtered([](const auto &path) { return path.extension().string() == ".jpg"; }),
                std::back_inserter(filenames));

    std::random_device seed;
    std::mt19937 engine(seed());

    std::mutex mutex;
    boost::progress_display progress_display(variables_map["num_data"].as<int>());

    std::vector<std::thread> threads;
    auto num_threads = std::thread::hardware_concurrency();
    auto num_data = variables_map["num_data"].as<int>() / num_threads;

    for (auto i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (auto j = num_data * i; j < num_data * (i + 1); ++j) {
                boost::gil::rgb8_image_t multi_image(variables_map["image_width"].as<int>(), variables_map["image_height"].as<int>());
                boost::gil::fill_pixels(boost::gil::view(multi_image), boost::gil::rgb8_pixel_t(0, 0, 0));
                std::vector<std::pair<std::string, boost::geometry::model::box<boost::geometry::model::d2::point_xy<int>>>> words;

                auto seq_len = std::uniform_int_distribution<int>(1, variables_map["seq_lens"].as<std::vector<int>>()[0])(engine);
                while (words.size() < seq_len) {
                    if (![&]() {
                            for (auto k = 0; k < variables_map["num_retries"].as<int>(); ++k) {
                                const auto &filename = filenames[std::uniform_int_distribution<int>(0, filenames.size() - 1)(engine)];

                                auto string = filename.stem().string();
                                std::smatch match;
                                if (!std::regex_match(string, match, std::regex(R"([0-9]+_([0-9A-Za-z]+))"))) {
                                    continue;
                                }
                                if (match.str(1).size() > variables_map["seq_lens"].as<std::vector<int>>()[1]) {
                                    continue;
                                }

                                boost::gil::rgb8_image_t image;
                                try {
                                    boost::gil::read_image(filename.string(), image, boost::gil::jpeg_tag());
                                } catch (...) {
                                    continue;
                                }
                                if (image.height() > multi_image.height() || image.width() > multi_image.width()) {
                                    continue;
                                }

                                if ([&]() {
                                        for (auto l = 0; l < variables_map["num_retries"].as<int>(); ++l) {
                                            auto dx = std::uniform_int_distribution<int>(0, multi_image.width() - image.width())(engine);
                                            auto dy = std::uniform_int_distribution<int>(0, multi_image.height() - image.height())(engine);
                                            boost::geometry::model::box<boost::geometry::model::d2::point_xy<int>> box(
                                                boost::geometry::model::d2::point_xy<int>(dx, dy),
                                                boost::geometry::model::d2::point_xy<int>(dx + image.width(), dy + image.height()));

                                            if (boost::algorithm::all_of(
                                                    words, [&](const auto &word) { return boost::geometry::disjoint(word.second, box); })) {
                                                boost::gil::copy_pixels(
                                                    boost::gil::view(image),
                                                    boost::gil::subimage_view(boost::gil::view(multi_image), dx, dy, image.width(), image.height()));
                                                words.emplace_back(match.str(1), box);
                                                return true;
                                            }
                                        }
                                        return false;
                                    }()) {
                                    return true;
                                }
                            }
                            return false;
                        }()) {
                        break;
                    }
                }

                boost::sort(words, [](const auto &word1, const auto &word2) {
                    return (word1.second.min_corner().y() < word2.second.min_corner().y()) ||
                           ((word1.second.min_corner().y() == word2.second.min_corner().y()) &&
                            (word1.second.min_corner().x() < word2.second.min_corner().x()));
                });

                auto stem = boost::accumulate(words, std::to_string(j), [](const auto &acc, const auto &word) { return acc + "_" + word.first; });
                boost::gil::write_view(variables_map["output_directory"].as<std::string>() + "/" + stem + ".jpg", boost::gil::view(multi_image),
                                       boost::gil::jpeg_tag());

                mutex.lock();
                ++progress_display;
                mutex.unlock();
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    return 0;
}
