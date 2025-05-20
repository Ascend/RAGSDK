/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_FILESYSTEM_H
#define ATB_SPEED_UTILS_FILESYSTEM_H
#include <string>
#include <vector>

namespace atb_speed {
class FileSystem {
public:
    static void GetDirChildItems(const std::string &dirPath, std::vector<std::string> &itemPaths);
    static void GetDirChildFiles(const std::string &dirPath, std::vector<std::string> &filePaths);
    static void GetDirChildDirs(const std::string &dirPath, std::vector<std::string> &dirPaths);
    static bool Exists(const std::string &path);
    static bool IsDir(const std::string &path);
    static std::string Join(const std::vector<std::string> &paths);
    static int64_t FileSize(const std::string &filePath);
    static std::string BaseName(const std::string &filePath);
    static std::string DirName(const std::string &path);
    static bool IsPathValid(const std::string &filePath);
    static bool ReadFile(const std::string &filePath, char *buffer, uint64_t bufferSize);
    static void DeleteFile(const std::string &filePath);
    static bool Rename(const std::string &filePath, const std::string &newFilePath);
    static bool MakeDir(const std::string &dirPath, int mode);
    static bool Makedirs(const std::string &dirPath, const mode_t mode);

private:
    static void GetDirChildItemsImpl(const std::string &dirPath, bool matchFile, bool matchDir,
                                     std::vector<std::string> &itemPaths);
};
} // namespace atb_speed
#endif