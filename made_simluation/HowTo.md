# 怎么跑v2xsim上的攻击/防御实验

1. 主函数 test_attack_det.py
    参数：
        --resume [ 模型文件 ]
        --detection_method [ none | match_cost_v2 | residual_autoencoder_v2 | raw_autoencoder | multi-test-raev2 | match_cost_unsupervised | robosac ] 其余参数是实验性质的，已经不用了
        --attack [攻击配置yaml文件，在config/attack/里，文件名命名方式 N{attacker数}_E{epsilon大小}_S{攻击step数}，带有shift的攻击有额外后缀，_shift表示加shift，_sep表示分区域加shift和抹除]
        --log 记录结果
        --logpath [ 记录结果的路径 ]
        --att_subpath [ logpath下攻击方式的子路径，使用multi-test加载已保存的match cost和reconstruction loss时，会利用该路径找记录文件 ]
2. 实验原始结果记录在 `/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments` 中
    其中 attack/ 是不同攻击参数的结果
        match_cost_v2/ 是match cost的结果
        residual_ae_v2/ 是residual autoencoder的结果
        raw_ae/ 是raw autoencoder的结果
        match_cost_raev2 是multi test的结果
        match_cost_unsupervised 是match_cost_unsupervised的结果
    分区攻击的结果在每个防御方式目录下的 `shift/` 里，纯pgd攻击的在 `gt/` 里，
3. 对抗训练的结果
    shift+pgd攻击下的对抗训练模型及在不同攻击下的表现 `/DB/data/yanghengzhao/adversarial/DAMC/yanghengzhao/disco-net/experiments/AT_shift/N01_E1e-1_S10_shift`
    pgd攻击的的对抗训练模型 `/DB/data/yanghengzhao/adversarial/DiscoNet/AT/disco/epoch_100.pth`，不同攻击下的表现在`experiments/AT_E1e-01_S1/` 里

4. 各种实验的参考脚本在`scripts/`里