#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/gil.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::literals::string_literals;

int main(int argc, char* argv[]) {
    boost::program_options::options_description options_description("option");
    options_description.add_options()("input_directory", boost::program_options::value<std::string>(), "directory of input data")(
        "output_directory", boost::program_options::value<std::string>(), "directory of output data")(
        "image_width", boost::program_options::value<int>()->default_value(256), "width of image that will be generated")(
        "image_height", boost::program_options::value<int>()->default_value(256), "height of image that will be generated")(
        "sequence_lengths", boost::program_options::value<std::vector<int>>()->multitoken()->default_value({4, 10}, "4, 10"), "sequence lengths of texts")(
        "num_data", boost::program_options::value<int>()->default_value(1000000), "number of data that will be generated")(
        "num_retries", boost::program_options::value<int>()->default_value(100), "number of retries for locating bounding box");

    boost::program_options::variables_map variables_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options_description), variables_map);
    boost::program_options::notify(variables_map);

    std::vector<boost::filesystem::path> filenames;
    for (const auto& entry : boost::make_iterator_range(boost::filesystem::recursive_directory_iterator(variables_map["input_directory"].as<std::string>()), {})) {
        if (entry.path().extension().string() == ".jpg") {
            filenames.emplace_back(entry.path());
        }
    }

    std::random_device seed;
    std::mt19937 engine(seed());

    auto num_threads = std::thread::hardware_concurrency();
    auto num_data = variables_map["num_data"].as<int>() / num_threads;

    std::vector<std::thread> threads;
    for (auto i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            boost::progress_display progress_display(num_data);
            for (auto j = num_data * i; j < num_data * (i + 1); ++j, ++progress_display) {
                boost::gil::rgb8_image_t multi_image(variables_map["image_width"].as<int>(), variables_map["image_height"].as<int>());
                boost::gil::fill_pixels(boost::gil::view(multi_image), boost::gil::rgb8_pixel_t(0, 0, 0));
                std::vector<std::pair<std::string, boost::geometry::model::box<boost::geometry::model::d2::point_xy<int>>>> strings;

                auto sequence_length = std::uniform_int_distribution<int>(1, variables_map["sequence_lengths"].as<std::vector<int>>()[0])(engine);
                if (!([&]() {
                        for (auto k = 0; k < variables_map["num_retries"].as<int>(); ++k) {
                            const auto& filename = filenames[std::uniform_int_distribution<int>(0, filenames.size() - 1)(engine)];

                            std::smatch match;
                            std::regex regex(R"([0-9]+_([0-9A-Za-z]*))");
                            if (!std::regex_match(filename.stem().string(), match, regex) ||
                                match[1].str().size() > variables_map["sequence_lengths"].as<std::vector<int>>()[1])
                                continue;

                            boost::gil::rgb8_image_t image;
                            boost::gil::read_image(filename.string(), image, boost::gil::jpeg_tag());

                            if (image.height() > multi_image.height() || image.width() > multi_image.width()) continue;

                            auto dx = std::uniform_int_distribution<int>(0, multi_image.width() - image.width())(engine);
                            auto dy = std::uniform_int_distribution<int>(0, multi_image.height() - image.height())(engine);
                            boost::geometry::model::box<boost::geometry::model::d2::point_xy<int>> box(
                                boost::geometry::model::d2::point_xy<int>(dx, dy), boost::geometry::model::d2::point_xy<int>(dx + image.width(), dy + image.height()));

                            if (std::all_of(strings.begin(), strings.end(), [&](const auto& string) { return boost::geometry::disjoint(string.second, box); })) {
                                boost::gil::copy_pixels(boost::gil::view(image),
                                                        boost::gil::subimage_view(boost::gil::view(multi_image), dx, dy, image.width(), image.height()));
                                strings.emplace_back(match[1].str(), box);

                                if (strings.size() == sequence_length) return true;
                            }
                        }

                        std::cout << "FFFFFFF" << std::endl;

                        return false;
                    }()))
                    continue;

                std::sort(strings.begin(), strings.end(), [](const auto& string1, const auto& string2) {
                    return (string1.second.min_corner().y() < string2.second.min_corner().y()) ||
                           ((string1.second.min_corner().y() == string2.second.min_corner().y()) && (string1.second.min_corner().x() < string2.second.min_corner().x()));
                });

                auto stem =
                    std::accumulate(strings.begin(), strings.end(), std::to_string(j), [](const auto& acc, const auto& string) { return acc + "_" + string.first; });
                boost::gil::write_view(variables_map["output_directory"].as<std::string>() + "/" + stem + ".jpg", boost::gil::view(multi_image), boost::gil::jpeg_tag());
            }
        });
    }

    for (auto& thread : threads) thread.join();

    return 0;
}