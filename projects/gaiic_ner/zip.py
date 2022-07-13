
def zip_files(dirpath='.', outFullName='code_data.zip'):
    import zipfile
    import os
    import time
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    print('开始压缩')
    start = time.time()
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            if filename == 'submit_result.py':
                continue
            if 'zip' in filename:
                continue
            if 'git' in filename:
                continue
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()
    end = time.time()
    print(f'压缩完成，用时：{end-start}')


if __name__ == '__main__':
    zip_files(dirpath='.', outFullName='code_data.zip')