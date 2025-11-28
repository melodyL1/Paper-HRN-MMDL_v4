
# 常用工具
def string_to_int_list(string):
    # 使用逗号分隔字符串
    str_list = string.split(',')
    # 初始化整数列表
    int_list = []

    # 遍历字符串列表
    for item in str_list:
        # 如果字符串表示true，则转换为1
        if item.lower() == "true":
            int_list.append(1)
        # 如果字符串表示false，则转换为0
        elif item.lower() == "false":
            int_list.append(0)
        # 否则，假设字符串是数字，直接转换为整数
        else:
            int_list.append(int(item))
    return int_list