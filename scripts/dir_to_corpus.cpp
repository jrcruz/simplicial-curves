#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <experimental/filesystem>



constexpr const char* CORPUS_PATH       = "corpus.dat";
constexpr const char* VOCABULARY_PATH   = "vocabulary.txt";
constexpr int DEFAULT_STOPWORD_COUNT    = 100;
constexpr int DEFAULT_MININUM_FREQUENCY = 4;
constexpr int MINIMUM_LENGTH            = 2;
constexpr int MAXIMUM_LENGTH            = 100;


enum class OutputSetting : char { NUMBER, TEXT } output_setting;
struct Term { int word; int count; bool seen_in_this_document; };

std::unordered_map<std::string, Term> lexicon;
std::unordered_set<std::string> stoplist;
int total_count = 0;



bool
acceptableLength(const std::string& word)
{
    return word.size() > MINIMUM_LENGTH and word.size() < MAXIMUM_LENGTH;
}


std::vector<std::string>
splitOnPunct(const std::string& word)
{
    std::vector<std::string> split_vector;
    std::string tmp;
    for (char c : word) {
        if (std::ispunct(c) != 0) {
            if (acceptableLength(tmp)) {
                split_vector.emplace_back(tmp);
                tmp.clear();
            }
        }
        else {
            tmp.push_back(std::tolower(c));
        }
    }
    if (acceptableLength(tmp)) {
        split_vector.emplace_back(tmp);
    }
    return split_vector;
}


void
initializeStoplist(const std::string& doc_dir, int stopword_count, int minimum_frequency)
{
    std::unordered_map<std::string, int> words_and_freqs;
    // The files are not sorted as they are in the directory.
    std::ifstream file_paths(doc_dir);
    std::string file_entry;
    while (file_paths >> file_entry) {
        std::ifstream file{file_entry};
        std::string raw_word;
        while (file >> raw_word) {
            for (const std::string& word : splitOnPunct(raw_word)) {
                ++words_and_freqs[word];
            }
        }
    }
    std::vector<std::pair<std::string, int>> freq_vec;
    std::move(std::begin(words_and_freqs), std::end(words_and_freqs),
              std::back_inserter(freq_vec));
    std::sort(std::begin(freq_vec), std::end(freq_vec),
        [](const auto& p1, const auto& p2) {
            return p1.second > p2.second;
        });
    // Remove most frequent words from corpus.
    for (int j = 0; j < stopword_count; ++j) {
        stoplist.emplace(std::move(freq_vec[j].first));
    }
    // Remove least frequent words (appear < minimum frequency) from corpus.
    for (auto from_last = std::rbegin(freq_vec), from_first = std::rend(freq_vec);
         from_last != from_first and from_last->second < minimum_frequency;
         ++from_last)
    {
        stoplist.emplace(std::move(from_last->first));
    }
}


int
readSingleFile(const std::string& filename, std::ofstream& vocabulary)
{
    std::ifstream file{filename};
    std::string raw_word;
    int current_unique_count = 0;
    // Read entire file.
    while (file >> raw_word) {
        for (const std::string& stripped : splitOnPunct(raw_word)) {
            // Skip word if it's a stopword.
            if (stoplist.find(stripped) != std::end(stoplist)) {
                continue;
            }
            // New word.
            if (lexicon.find(stripped) == std::end(lexicon)) {
                lexicon[stripped] = {total_count++, 1, true};
                vocabulary << stripped << '\n';
                ++current_unique_count;
            }
            // Old word, already seen.
            else {
                ++lexicon[stripped].count;
                // The word is already in the lexicon (from a previous file)
                // but it's the first time we see it in _this_ document.
                if (not lexicon[stripped].seen_in_this_document) {
                    lexicon[stripped].seen_in_this_document = true;
                    ++current_unique_count;
                }
            }
        }
    }
    return current_unique_count;
}


void
writeFileTerms(std::ofstream& data_file, int current_count)
{
    // '<n distinct terms> <index in vocabulary>:<count in corpus>'.
    data_file << current_count << ' ';
    for (auto& lexicon_data : lexicon) {
        if (lexicon_data.second.count == 0) {
            continue;
        }
        if (output_setting == OutputSetting::TEXT) {
            data_file << lexicon_data.first << ' ';
        }
        else {
            data_file << lexicon_data.second.word  << ':'
                      << lexicon_data.second.count << ' ';
        }
        // Reset the lexicon counts because we're moving to a new file.
        lexicon_data.second.count = 0;
        // The words stay in the lexicon but we need to make sure that the
        // next iteration will consider each already known word as a first
        // time visualization for purpose of counting the unique terms in
        // the next document.
        lexicon_data.second.seen_in_this_document = false;
    }
    data_file << '\n';
}


// Compile with '-std=c++1z <file> -lstdc++fs'. Needs to be the experimental version.
int
main(int argc, const char* argv[])
{
    if (argc < 3) {
        std::cout << "usage: ./dir_to_corpus <dir_listing> {text | number} "
        "[<minimum frequency (exclusive)>] [<num stopwords>] "
        "[<corpus output>] [<vocabulary output>]\n";
        std::exit(EXIT_FAILURE);
    }
    if (not std::experimental::filesystem::exists(argv[1])) {
        std::cout << "Directory does not exist, exiting.\n";
        std::exit(EXIT_FAILURE);
    }
    if (std::string{argv[2]} == "text") {
        output_setting = OutputSetting::TEXT;
    }
    else if (std::string{argv[2]} == "number") {
        output_setting = OutputSetting::NUMBER;
    }
    else {
        std::cout << "usage: ./dir_to_corpus <dir_listing> {text | number} "
        "[<mininum frequency (exclusive)>] [<num stopwords>] "
        "[<corpus output>] [<vocabulary output>]\n";
        std::exit(EXIT_FAILURE);
    }
    int mininum_frequency = argc >= 4 ? std::stoi(argv[3]) : DEFAULT_MININUM_FREQUENCY;
    int stopword_count    = argc >= 5 ? std::stoi(argv[4]) : DEFAULT_STOPWORD_COUNT;

    std::cout << "Creating stoplist.\n";
    initializeStoplist(argv[1], stopword_count, mininum_frequency);

    std::ofstream data_file      {argc >= 6 ? argv[5] : CORPUS_PATH};
    std::ofstream vocabulary_file{argc >= 7 ? argv[6] : VOCABULARY_PATH};
    // The files are not sorted as they are in the directory.
    std::cout << "Reading files.\n";

    std::ifstream file_paths(argv[1]);
    std::string file_entry;
    while (file_paths >> file_entry) {
        int current_count = readSingleFile(file_entry, vocabulary_file);
        writeFileTerms(data_file, current_count);
    }
    std::cout << "Vocabulary length: " << total_count << '\n';
}
