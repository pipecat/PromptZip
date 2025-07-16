from compressor import QwenCompressor

PROMPT = """
马自达（Mazda），全称马自达株式会社（日文：マツダ株式会社，英文：Mazda Motor Corporation），是一家在东京证交所（TYO）上市的日本跨国汽车制造商，2020年度《财富》世界五百强排名第400位。 [1-3]。
公司成立于1920年，初称东洋コルク工业株式会社，1984年改为现名。 [4]公司总部位于日本国广岛县安芸郡，现任社长丸本明（Marumoto Akira）。 [1]
2019至2020财年马自达销量约142万台，均由Mazda单一品牌贡献，最大市场为北美市场，年销量约40万台。 [5]2019年马自达在华销量约23万台，全部由合资企业长安马自达和一汽马自达贡献。 [6]
2020年马自达全球销量约124万台，最大市场为美国市场，销量约28万台；中国市场其次，销量约21万台。 [22]
2024年7月16日消息，在日本国土交通省表示马自达汽车符合标准后，马自达汽车将自7月18日起恢复生产两款被暂停生产的车型。 [52]
2025年1月21日，据外媒 Drive 报道，马自达 6 系列车型将退出澳洲市场。
截2021年，马自达乘用车全球产品序列包括：SUV系列CX-3、CX-30、cx-50、CX-4、CX-5、CX-8、CX-9、MX-30；轿车系列Mazda2、Mazda3（长安马自达又称“昂克塞拉”）、Mazda6（一汽轿车称“阿特兹”）；跑车MX-5；马自达EZ-6。 [23]
"""

if __name__ == "__main__":
    compressor = QwenCompressor()
    compressor.load_model("Qwen/Qwen3-0.6B")
    compressed_prompt = compressor.compress(PROMPT)
    print(compressed_prompt)