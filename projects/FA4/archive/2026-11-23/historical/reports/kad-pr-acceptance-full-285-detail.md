# KDA vs PR3109 全场景逐项明细

范围：Forward 165 + Decode 120；Backward 排除。`speedup = PR mean / KDA mean`，大于 1 表示 KDA 更快。完整字段与场景参数见同目录 CSV。

|#|阶段|场景|Owner|KDA配置|PR ms|KDA ms|加速比|变化|最大绝对误差|正确|
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
|1|forward|bfloat16-d96-gqa-s512-global|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.322285|0.303936|1.060371|+6.037%|0|True|
|2|forward|float16-d128-mqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.911635|0.899597|1.013382|+1.338%|0|True|
|3|forward|float16-d128-mha-s512-causal|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.328461|0.312531|1.050970|+5.097%|0|True|
|4|forward|bfloat16-d128-mqa-s512-causal|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.258816|0.249670|1.036631|+3.663%|0|True|
|5|forward|bfloat16-d96-mqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.209562|0.202483|1.034958|+3.496%|0|True|
|6|forward|bfloat16-d64-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.587526|1.582003|1.003491|+0.349%|0|True|
|7|forward|bfloat16-d192-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|2.191206|2.184544|1.003050|+0.305%|0|True|
|8|forward|bfloat16-d128-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.876090|2.871482|1.001605|+0.160%|0|True|
|9|forward|bfloat16-d96-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.513421|0.505190|1.016292|+1.629%|0|True|
|10|forward|float16-d128-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.929376|2.934752|0.998168|-0.183%|0|True|
|11|forward|bfloat16-d192-mha-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.409702|0.405107|1.011343|+1.134%|0|True|
|12|forward|bfloat16-d256-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|3.279987|3.268269|1.003586|+0.359%|0|True|
|13|forward|bfloat16-d64-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.918413|2.921811|0.998837|-0.116%|0|True|
|14|forward|bfloat16-d192-mqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|1.186368|1.176378|1.008492|+0.849%|0|True|
|15|forward|float16-d128-mqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.260928|0.253613|1.028844|+2.884%|0|True|
|16|forward|float16-d128-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.537850|0.526925|1.020733|+2.073%|0|True|
|17|forward|bfloat16-d192-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|2.242982|2.232890|1.004520|+0.452%|0|True|
|18|forward|bfloat16-d128-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.567994|1.565312|1.001713|+0.171%|0|True|
|19|forward|bfloat16-d256-mha-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.513926|0.504954|1.017770|+1.777%|0|True|
|20|forward|bfloat16-d96-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.235200|1.227603|1.006188|+0.619%|0|True|
|21|forward|float16-d128-gqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.545965|0.539142|1.012654|+1.265%|0|True|
|22|forward|bfloat16-d96-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.404083|0.400851|1.008063|+0.806%|0|True|
|23|forward|bfloat16-d128-gqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.889267|0.884890|1.004947|+0.495%|0|True|
|24|forward|bfloat16-d128-mqa-s512-local1024|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.259078|0.250208|1.035452|+3.545%|0|True|
|25|forward|bfloat16-d64-mqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.317683|0.310400|1.023464|+2.346%|0|True|
|26|forward|bfloat16-d192-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|2.186387|2.198739|0.994382|-0.562%|0|True|
|27|forward|bfloat16-d256-gqa-s512-causal|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.297062|0.277613|1.070060|+7.006%|0|True|
|28|forward|bfloat16-d128-gqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.546861|0.543552|1.006087|+0.609%|0|True|
|29|forward|bfloat16-d192-mqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.332960|0.326221|1.020658|+2.066%|0|True|
|30|forward|bfloat16-d96-mha-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.296083|0.289600|1.022387|+2.239%|0|True|
|31|forward|float16-d128-mqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.483117|0.469312|1.029415|+2.941%|0|True|
|32|forward|bfloat16-d96-mha-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.476768|0.467277|1.020312|+2.031%|0|True|
|33|forward|float16-d128-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.600198|1.590739|1.005946|+0.595%|0|True|
|34|forward|float16-d128-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.528429|0.516518|1.023059|+2.306%|0|True|
|35|forward|float16-d128-mha-s512-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.348838|0.347130|1.004923|+0.492%|0|True|
|36|forward|float16-d128-gqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.466419|0.447526|1.042216|+4.222%|0|True|
|37|forward|bfloat16-d128-mha-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.549549|0.543130|1.011819|+1.182%|0|True|
|38|forward|float16-d128-gqa-s512-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.356890|0.340608|1.047802|+4.780%|0|True|
|39|forward|bfloat16-d192-mqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.604858|0.592410|1.021013|+2.101%|0|True|
|40|forward|bfloat16-d256-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.595578|0.593510|1.003483|+0.348%|0|True|
|41|forward|bfloat16-d128-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.523373|0.519392|1.007664|+0.766%|0|True|
|42|forward|bfloat16-d96-mqa-s512-global|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.272653|0.254458|1.071506|+7.151%|0|True|
|43|forward|float16-d128-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.553050|1.552845|1.000132|+0.013%|0|True|
|44|forward|bfloat16-d128-gqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.346144|0.344282|1.005410|+0.541%|0|True|
|45|forward|bfloat16-d256-mha-s512-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.374176|0.363315|1.029894|+2.989%|0|True|
|46|forward|bfloat16-d256-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.583782|0.578272|1.009529|+0.953%|0|True|
|47|forward|float16-d128-mqa-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.533018|0.525101|1.015077|+1.508%|0|True|
|48|forward|bfloat16-d256-mha-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.953248|0.952653|1.000625|+0.062%|0|True|
|49|forward|bfloat16-d96-gqa-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.403392|0.398413|1.012498|+1.250%|0|True|
|50|forward|bfloat16-d128-mha-s512-local1024|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.314291|0.309037|1.017003|+1.700%|0|True|
|51|forward|bfloat16-d192-mqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.725171|0.712704|1.017493|+1.749%|0|True|
|52|forward|bfloat16-d128-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.561626|1.548806|1.008277|+0.828%|0|True|
|53|forward|float16-d128-mha-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.547814|0.544563|1.005970|+0.597%|0|True|
|54|forward|bfloat16-d96-mha-s512-local1024|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.276774|0.264518|1.046333|+4.633%|0|True|
|55|forward|bfloat16-d192-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|4.134259|4.157325|0.994452|-0.555%|0|True|
|56|forward|bfloat16-d96-mqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.665466|0.663456|1.003029|+0.303%|0|True|
|57|forward|bfloat16-d192-gqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|1.230093|1.216973|1.010781|+1.078%|0|True|
|58|forward|bfloat16-d128-mqa-varlen-causal|pr-fallback|`PR fallback (no KDA config)`|1.142829|1.138080|1.004173|+0.417%|0|True|
|59|forward|bfloat16-d64-mha-s2048-global|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.866835|0.862573|1.004941|+0.494%|0|True|
|60|forward|bfloat16-d64-gqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.252512|0.246221|1.025551|+2.555%|0|True|
|61|forward|bfloat16-d96-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.406662|0.399565|1.017763|+1.776%|0|True|
|62|forward|bfloat16-d128-gqa-varlen-causal|pr-fallback|`PR fallback (no KDA config)`|1.147962|1.139923|1.007052|+0.705%|0|True|
|63|forward|bfloat16-d128-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.561549|1.553510|1.005174|+0.517%|0|True|
|64|forward|bfloat16-d64-mha-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.343898|0.332506|1.034261|+3.426%|0|True|
|65|forward|float16-d128-mha-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.314221|0.307437|1.022066|+2.207%|0|True|
|66|forward|float16-d128-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.545395|0.529453|1.030111|+3.011%|0|True|
|67|forward|bfloat16-d256-mqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.291046|0.278950|1.043363|+4.336%|0|True|
|68|forward|bfloat16-d64-gqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.866272|0.855078|1.013091|+1.309%|0|True|
|69|forward|bfloat16-d64-mha-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.466528|0.458368|1.017802|+1.780%|0|True|
|70|forward|bfloat16-d256-mqa-s512-local1024|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.294566|0.285939|1.030171|+3.017%|0|True|
|71|forward|float16-d128-gqa-s512-causal|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.267770|0.262861|1.018675|+1.867%|0|True|
|72|forward|float16-d128-mqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.355514|0.337824|1.052363|+5.236%|0|True|
|73|forward|bfloat16-d192-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|4.176109|4.206566|0.992760|-0.724%|0|True|
|74|forward|float16-d128-mqa-s512-causal|kad-direct|`{"num_splits":1,"num_threads":64,"pack_gqa":true,"tile_m":64,"tile_n":64}`|0.261638|0.245728|1.064748|+6.475%|0|True|
|75|forward|bfloat16-d96-mha-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.692358|0.682106|1.015031|+1.503%|0|True|
|76|forward|bfloat16-d64-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.960262|2.965235|0.998323|-0.168%|0|True|
|77|forward|bfloat16-d64-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.497376|0.497440|0.999871|-0.013%|0|True|
|78|forward|bfloat16-d192-gqa-s512-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.351398|0.333530|1.053575|+5.357%|0|True|
|79|forward|bfloat16-d192-gqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.724691|0.719802|1.006793|+0.679%|0|True|
|80|forward|bfloat16-d256-gqa-s2048-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.944166|0.936710|1.007960|+0.796%|0|True|
|81|forward|bfloat16-d256-mqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.381779|0.372794|1.024103|+2.410%|0|True|
|82|forward|bfloat16-d64-mha-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.540640|0.530733|1.018667|+1.867%|0|True|
|83|forward|bfloat16-d192-mqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.481414|0.479174|1.004675|+0.467%|0|True|
|84|forward|bfloat16-d256-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.696365|1.775859|0.955236|-4.476%|0|True|
|85|forward|bfloat16-d128-mqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.853216|0.845581|1.009030|+0.903%|0|True|
|86|forward|bfloat16-d128-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.532550|0.520384|1.023380|+2.338%|0|True|
|87|forward|bfloat16-d256-gqa-s512-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.373875|0.370784|1.008337|+0.834%|0|True|
|88|forward|bfloat16-d256-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|3.208826|3.218163|0.997098|-0.290%|0|True|
|89|forward|bfloat16-d96-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.272013|2.271661|1.000155|+0.015%|0|True|
|90|forward|bfloat16-d96-mqa-s2048-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.347539|0.334323|1.039531|+3.953%|0|True|
|91|forward|bfloat16-d96-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.168314|1.161114|1.006201|+0.620%|0|True|
|92|forward|bfloat16-d64-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.499130|0.488294|1.022190|+2.219%|0|True|
|93|forward|bfloat16-d64-gqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.333562|0.330016|1.010744|+1.074%|0|True|
|94|forward|bfloat16-d128-mha-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.314118|0.307571|1.021287|+2.129%|0|True|
|95|forward|bfloat16-d128-gqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.269651|0.258726|1.042225|+4.223%|0|True|
|96|forward|bfloat16-d64-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.560384|1.550829|1.006161|+0.616%|0|True|
|97|forward|bfloat16-d64-gqa-s2048-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.437274|0.424678|1.029658|+2.966%|0|True|
|98|forward|float16-d128-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.982822|2.991354|0.997148|-0.285%|0|True|
|99|forward|bfloat16-d96-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.246470|2.242112|1.001944|+0.194%|0|True|
|100|forward|bfloat16-d192-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.753562|0.740205|1.018045|+1.804%|0|True|
|101|forward|float16-d128-mha-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.472058|0.461094|1.023776|+2.378%|0|True|
|102|forward|bfloat16-d64-mha-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.305082|0.303072|1.006631|+0.663%|0|True|
|103|forward|bfloat16-d128-gqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.271059|0.265331|1.021588|+2.159%|0|True|
|104|forward|bfloat16-d192-mha-s512-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.408781|0.402432|1.015776|+1.578%|0|True|
|105|forward|bfloat16-d192-gqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.471258|0.464365|1.014844|+1.484%|0|True|
|106|forward|bfloat16-d256-mha-s512-causal|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.315462|0.306003|1.030912|+3.091%|0|True|
|107|forward|float16-d128-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.553562|1.551539|1.001303|+0.130%|0|True|
|108|forward|bfloat16-d256-gqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.499194|0.494067|1.010376|+1.038%|0|True|
|109|forward|bfloat16-d64-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.954618|2.955014|0.999866|-0.013%|0|True|
|110|forward|bfloat16-d96-mqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.206502|0.199597|1.034598|+3.460%|0|True|
|111|forward|bfloat16-d96-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.242957|2.244339|0.999384|-0.062%|0|True|
|112|forward|float16-d128-mha-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.902669|0.887994|1.016526|+1.653%|0|True|
|113|forward|bfloat16-d256-mha-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.588077|0.579149|1.015416|+1.542%|0|True|
|114|forward|bfloat16-d64-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.543834|1.540346|1.002264|+0.226%|0|True|
|115|forward|bfloat16-d64-mqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.241805|0.240851|1.003959|+0.396%|0|True|
|116|forward|bfloat16-d96-gqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.347558|0.338112|1.027939|+2.794%|0|True|
|117|forward|bfloat16-d128-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.855443|2.864371|0.996883|-0.312%|0|True|
|118|forward|bfloat16-d192-gqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.625101|0.589440|1.060499|+6.050%|0|True|
|119|forward|bfloat16-d128-gqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.451955|0.444909|1.015838|+1.584%|0|True|
|120|forward|bfloat16-d192-mha-s2048-global|pr-fallback|`PR fallback (no KDA config)`|1.214317|1.208493|1.004819|+0.482%|0|True|
|121|forward|bfloat16-d192-mha-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.637517|0.634784|1.004305|+0.431%|0|True|
|122|forward|bfloat16-d128-mha-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.351392|0.346170|1.015086|+1.509%|0|True|
|123|forward|bfloat16-d64-mqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.854342|0.845862|1.010025|+1.003%|0|True|
|124|forward|bfloat16-d128-mqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.531917|0.528813|1.005870|+0.587%|0|True|
|125|forward|bfloat16-d256-gqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.577267|0.574214|1.005316|+0.532%|0|True|
|126|forward|bfloat16-d128-mqa-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.340909|0.333395|1.022537|+2.254%|0|True|
|127|forward|bfloat16-d128-mha-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.925414|2.920646|1.001633|+0.163%|0|True|
|128|forward|bfloat16-d256-gqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.292832|0.285984|1.023945|+2.395%|0|True|
|129|forward|bfloat16-d192-mha-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.738618|0.739930|0.998227|-0.177%|0|True|
|130|forward|bfloat16-d64-gqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.513837|0.511072|1.005410|+0.541%|0|True|
|131|forward|bfloat16-d128-mqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.442656|0.433850|1.020298|+2.030%|0|True|
|132|forward|bfloat16-d64-mha-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.538995|0.527936|1.020948|+2.095%|0|True|
|133|forward|bfloat16-d96-mha-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.288486|0.284179|1.015157|+1.516%|0|True|
|134|forward|bfloat16-d256-mqa-s2048-global|kad-direct|`{"num_threads":64,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.939514|0.933210|1.006755|+0.676%|0|True|
|135|forward|bfloat16-d96-mha-s2048-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.458778|0.379149|1.210020|+21.002%|0|True|
|136|forward|bfloat16-d192-mha-s512-global|pr-fallback|`PR fallback (no KDA config)`|0.458957|0.457786|1.002558|+0.256%|0|True|
|137|forward|bfloat16-d64-mqa-s512-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.249094|0.237114|1.050528|+5.053%|0|True|
|138|forward|bfloat16-d128-mha-varlen-causal|pr-fallback|`PR fallback (no KDA config)`|1.154227|1.145677|1.007463|+0.746%|0|True|
|139|forward|bfloat16-d256-mqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|3.214150|3.217056|0.999097|-0.090%|0|True|
|140|forward|bfloat16-d96-mqa-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.403302|0.392352|1.027910|+2.791%|0|True|
|141|forward|bfloat16-d128-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.541485|0.529440|1.022750|+2.275%|0|True|
|142|forward|bfloat16-d96-gqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.210502|0.197958|1.063367|+6.337%|0|True|
|143|forward|bfloat16-d192-mqa-s512-local1024|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.332672|0.329286|1.010282|+1.028%|0|True|
|144|forward|bfloat16-d256-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.580205|0.691168|0.839456|-16.054%|0|True|
|145|forward|bfloat16-d64-mqa-s2048-causal|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.509050|0.499718|1.018673|+1.867%|0|True|
|146|forward|bfloat16-d96-gqa-s512-causal|kad-direct|`{"num_splits":1,"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.206771|0.194598|1.062553|+6.255%|0|True|
|147|forward|bfloat16-d256-mha-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.725107|1.717517|1.004419|+0.442%|0|True|
|148|forward|bfloat16-d192-mqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.695059|0.685504|1.013939|+1.394%|0|True|
|149|forward|bfloat16-d64-mqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.427078|0.417638|1.022603|+2.260%|0|True|
|150|forward|bfloat16-d96-mqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.167987|1.158010|1.008616|+0.862%|0|True|
|151|forward|bfloat16-d64-gqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.264870|0.253664|1.044178|+4.418%|0|True|
|152|forward|bfloat16-d256-mqa-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.499930|0.490080|1.020098|+2.010%|0|True|
|153|forward|bfloat16-d128-mha-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.882630|0.876301|1.007223|+0.722%|0|True|
|154|forward|bfloat16-d256-mqa-s2048-causal|pr-fallback|`PR fallback (no KDA config)`|0.574522|0.569600|1.008640|+0.864%|0|True|
|155|forward|bfloat16-d192-gqa-s512-causal|pr-fallback|`PR fallback (no KDA config)`|0.338867|0.334195|1.013980|+1.398%|0|True|
|156|forward|bfloat16-d64-mha-s512-causal|kad-direct|`{"num_splits":1,"num_threads":64,"pack_gqa":false,"tile_m":64,"tile_n":64}`|0.306611|0.296922|1.032634|+3.263%|0|True|
|157|forward|bfloat16-d256-gqa-s8192-causal|pr-fallback|`PR fallback (no KDA config)`|1.703347|1.702854|1.000289|+0.029%|0|True|
|158|forward|float16-d128-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|2.953613|2.956058|0.999173|-0.083%|0|True|
|159|forward|bfloat16-d192-gqa-s8192-global|pr-fallback|`PR fallback (no KDA config)`|4.201139|4.224301|0.994517|-0.548%|0|True|
|160|forward|bfloat16-d192-gqa-s8192-local1024|pr-fallback|`PR fallback (no KDA config)`|0.699424|0.691693|1.011177|+1.118%|0|True|
|161|forward|float16-d128-gqa-s2048-global|pr-fallback|`PR fallback (no KDA config)`|0.895238|0.891315|1.004402|+0.440%|0|True|
|162|forward|float16-d128-gqa-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.263059|0.259482|1.013787|+1.379%|0|True|
|163|forward|bfloat16-d256-mha-s512-local1024|pr-fallback|`PR fallback (no KDA config)`|0.315533|0.314746|1.002501|+0.250%|0|True|
|164|forward|bfloat16-d128-mha-s2048-local1024|pr-fallback|`PR fallback (no KDA config)`|0.463834|0.456525|1.016010|+1.601%|0|True|
|165|forward|bfloat16-d96-gqa-s2048-global|kad-direct|`{"num_threads":128,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.670310|0.659277|1.016736|+1.674%|0|True|
|166|decode|bf16-d256-h8-hk2-b128-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|1.004749|0.866163|1.159999|+16.000%|0|True|
|167|decode|bf16-d64-h32-hk8-b32-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.206394|0.163238|1.264369|+26.437%|0|True|
|168|decode|bf16-d256-h8-hk2-b128-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.265165|0.245696|1.079239|+7.924%|0|True|
|169|decode|bf16-d256-h8-hk2-b1-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.215930|0.138310|1.561196|+56.120%|0|True|
|170|decode|bf16-d256-h8-hk2-b32-dense-loguniform64to32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.396307|0.385555|1.027887|+2.789%|0|True|
|171|decode|bf16-d256-h8-hk2-b8-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.239661|0.168486|1.422434|+42.243%|0|True|
|172|decode|bf16-d64-h32-hk8-b8-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.281133|0.169190|1.661636|+66.164%|0|True|
|173|decode|bf16-d256-h8-hk2-b1-dense-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.190470|0.135213|1.408671|+40.867%|0|True|
|174|decode|bf16-d64-h32-hk8-b128-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.943443|0.821517|1.148416|+14.842%|0|True|
|175|decode|bf16-d64-h32-hk8-b8-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.335296|0.188736|1.776534|+77.653%|0|True|
|176|decode|bf16-d64-h32-hk8-b8-dense-uniform32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.565517|0.478451|1.181974|+18.197%|0|True|
|177|decode|bf16-d64-h32-hk8-b32-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.385990|0.289888|1.331516|+33.152%|0|True|
|178|decode|bf16-d64-h32-hk8-b128-paged256-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|1.241677|1.143104|1.086233|+8.623%|0|True|
|179|decode|bf16-d256-h8-hk2-b128-paged256-loguniform64to32768|kad-direct|`{"num_splits":4,"pack_gqa":null}`|1.236467|1.108186|1.115758|+11.576%|0|True|
|180|decode|bf16-d256-h8-hk2-b8-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.194893|0.130541|1.492965|+49.296%|0|True|
|181|decode|bf16-d128-h16-hk4-b128-paged256-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|1.241850|1.180333|1.052118|+5.212%|0|True|
|182|decode|bf16-d64-h32-hk8-b128-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.260909|0.242970|1.073833|+7.383%|0|True|
|183|decode|bf16-d128-h16-hk4-b1-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.190765|0.174413|1.093755|+9.375%|0|True|
|184|decode|bf16-d64-h32-hk8-b1-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.192864|0.160819|1.199260|+19.926%|0|True|
|185|decode|bf16-d256-h8-hk2-b8-dense-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.247827|0.198566|1.248082|+24.808%|0|True|
|186|decode|bf16-d64-h32-hk8-b8-paged256-uniform32768|kad-direct|`{"num_splits":4,"pack_gqa":null,"tile_m":16,"tile_n":64}`|0.565542|0.484166|1.168074|+16.807%|0|True|
|187|decode|bf16-d64-h32-hk8-b1-paged256-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.189299|0.130202|1.453893|+45.389%|0|True|
|188|decode|bf16-d256-h8-hk2-b32-dense-uniform512|kad-direct|`{"intra_wg_overlap":true,"mma_pv_is_rs":true,"num_splits":1,"pack_gqa":true,"tile_m":64,"tile_n":64}`|0.251296|0.134374|1.870118|+87.012%|0|True|
|189|decode|bf16-d128-h16-hk4-b128-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.167706|0.145613|1.151723|+15.172%|0|True|
|190|decode|bf16-d64-h32-hk8-b128-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null}`|5.797504|5.703251|1.016526|+1.653%|0|True|
|191|decode|bf16-d256-h8-hk2-b8-dense-uniform32768|kad-direct|`{"num_splits":4,"pack_gqa":null}`|0.557562|0.479283|1.163324|+16.332%|0|True|
|192|decode|bf16-d256-h8-hk2-b1-paged256-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.213619|0.198490|1.076224|+7.622%|0|True|
|193|decode|bf16-d256-h8-hk2-b128-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.250067|0.155213|1.611125|+61.112%|0|True|
|194|decode|bf16-d128-h16-hk4-b1-paged256-uniform4096|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.260230|0.166131|1.566415|+56.641%|0|True|
|195|decode|bf16-d64-h32-hk8-b128-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.258611|0.246848|1.047654|+4.765%|0|True|
|196|decode|bf16-d128-h16-hk4-b8-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.251322|0.193926|1.295964|+29.596%|0|True|
|197|decode|bf16-d64-h32-hk8-b128-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.173459|0.141984|1.221681|+22.168%|0|True|
|198|decode|bf16-d64-h32-hk8-b1-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.191699|0.152109|1.260277|+26.028%|0|True|
|199|decode|bf16-d128-h16-hk4-b128-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.928224|0.817824|1.134992|+13.499%|0|True|
|200|decode|bf16-d128-h16-hk4-b128-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.197702|0.171046|1.155841|+15.584%|0|True|
|201|decode|bf16-d64-h32-hk8-b128-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.168250|0.149530|1.125193|+12.519%|0|True|
|202|decode|bf16-d256-h8-hk2-b32-paged256-loguniform64to32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.461485|0.305120|1.512470|+51.247%|0|True|
|203|decode|bf16-d128-h16-hk4-b128-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.323750|0.297133|1.089582|+8.958%|0|True|
|204|decode|bf16-d128-h16-hk4-b1-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.324736|0.234560|1.384447|+38.445%|0|True|
|205|decode|bf16-d128-h16-hk4-b32-paged256-loguniform64to32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.397312|0.380678|1.043695|+4.369%|0|True|
|206|decode|bf16-d256-h8-hk2-b32-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|1.573568|1.523955|1.032555|+3.256%|0|True|
|207|decode|bf16-d256-h8-hk2-b128-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.173350|0.167802|1.033068|+3.307%|0|True|
|208|decode|bf16-d64-h32-hk8-b32-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.271763|0.248288|1.094548|+9.455%|0|True|
|209|decode|bf16-d128-h16-hk4-b8-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":32,"tile_n":64}`|0.201734|0.130330|1.547879|+54.788%|0|True|
|210|decode|bf16-d64-h32-hk8-b1-dense-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.184915|0.125421|1.474358|+47.436%|0|True|
|211|decode|bf16-d256-h8-hk2-b8-paged256-uniform4096|kad-direct|`{"num_splits":2,"pack_gqa":null}`|0.207341|0.163981|1.264421|+26.442%|0|True|
|212|decode|bf16-d256-h8-hk2-b8-dense-loguniform64to32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.348570|0.275910|1.263343|+26.334%|0|True|
|213|decode|bf16-d128-h16-hk4-b32-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.399424|0.295949|1.349639|+34.964%|0|True|
|214|decode|bf16-d128-h16-hk4-b1-dense-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.183302|0.178867|1.024796|+2.480%|0|True|
|215|decode|bf16-d256-h8-hk2-b32-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.380128|0.293254|1.296240|+29.624%|0|True|
|216|decode|bf16-d64-h32-hk8-b1-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":128,"tile_n":128}`|0.205907|0.131264|1.568649|+56.865%|0|True|
|217|decode|bf16-d64-h32-hk8-b128-dense-uniform4096|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.913574|0.831571|1.098612|+9.861%|0|True|
|218|decode|bf16-d256-h8-hk2-b8-paged256-loguniform64to32768|kad-direct|`{"num_splits":8,"pack_gqa":null}`|0.297274|0.264365|1.124483|+12.448%|0|True|
|219|decode|bf16-d256-h8-hk2-b32-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.340282|0.297363|1.144330|+14.433%|0|True|
|220|decode|bf16-d128-h16-hk4-b8-dense-loguniform64to32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.345088|0.270995|1.273410|+27.341%|0|True|
|221|decode|bf16-d128-h16-hk4-b32-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.203712|0.130144|1.565282|+56.528%|0|True|
|222|decode|bf16-d128-h16-hk4-b128-dense-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":64}`|1.288237|1.170285|1.100789|+10.079%|0|True|
|223|decode|bf16-d64-h32-hk8-b1-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":128,"tile_n":64}`|0.264314|0.188749|1.400346|+40.035%|0|True|
|224|decode|bf16-d64-h32-hk8-b32-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.387603|0.297312|1.303692|+30.369%|0|True|
|225|decode|bf16-d256-h8-hk2-b1-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.189683|0.180141|1.052972|+5.297%|0|True|
|226|decode|bf16-d128-h16-hk4-b128-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|5.844429|5.695123|1.026216|+2.622%|0|True|
|227|decode|bf16-d128-h16-hk4-b8-dense-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.252326|0.192864|1.308313|+30.831%|0|True|
|228|decode|bf16-d128-h16-hk4-b32-dense-loguniform64to32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.391405|0.377664|1.036384|+3.638%|0|True|
|229|decode|bf16-d64-h32-hk8-b32-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|1.616954|1.528794|1.057666|+5.767%|0|True|
|230|decode|bf16-d256-h8-hk2-b128-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null}`|5.824595|5.726368|1.017153|+1.715%|0|True|
|231|decode|bf16-d128-h16-hk4-b1-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.191494|0.126176|1.517677|+51.768%|0|True|
|232|decode|bf16-d128-h16-hk4-b8-dense-uniform32768|kad-direct|`{"num_splits":2,"pack_gqa":null}`|0.559923|0.485715|1.152781|+15.278%|0|True|
|233|decode|bf16-d64-h32-hk8-b32-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.164416|0.135123|1.216786|+21.679%|0|True|
|234|decode|bf16-d128-h16-hk4-b32-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.148013|0.126362|1.171343|+17.134%|0|True|
|235|decode|bf16-d128-h16-hk4-b8-paged256-loguniform64to32768|kad-direct|`{"num_splits":8,"pack_gqa":null}`|0.351878|0.268653|1.309789|+30.979%|0|True|
|236|decode|bf16-d128-h16-hk4-b1-dense-uniform4096|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.184832|0.156006|1.184772|+18.477%|0|True|
|237|decode|bf16-d64-h32-hk8-b1-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.191456|0.177005|1.081643|+8.164%|0|True|
|238|decode|bf16-d64-h32-hk8-b128-dense-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|1.288096|1.131603|1.138293|+13.829%|0|True|
|239|decode|bf16-d64-h32-hk8-b1-dense-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.183258|0.174976|1.047330|+4.733%|0|True|
|240|decode|bf16-d64-h32-hk8-b8-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":16,"tile_n":64}`|0.191942|0.156237|1.228535|+22.854%|0|True|
|241|decode|bf16-d128-h16-hk4-b32-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|1.611987|1.512506|1.065773|+6.577%|0|True|
|242|decode|bf16-d64-h32-hk8-b8-dense-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.239283|0.165261|1.447913|+44.791%|0|True|
|243|decode|bf16-d128-h16-hk4-b1-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.195174|0.124525|1.567354|+56.735%|0|True|
|244|decode|bf16-d64-h32-hk8-b8-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.183283|0.124339|1.474058|+47.406%|0|True|
|245|decode|bf16-d256-h8-hk2-b128-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null}`|5.728358|5.698195|1.005293|+0.529%|0|True|
|246|decode|bf16-d128-h16-hk4-b32-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.250630|0.230093|1.089258|+8.926%|0|True|
|247|decode|bf16-d64-h32-hk8-b32-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|1.614976|1.510554|1.069129|+6.913%|0|True|
|248|decode|bf16-d64-h32-hk8-b128-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|6.068051|5.706202|1.063413|+6.341%|0|True|
|249|decode|bf16-d256-h8-hk2-b128-dense-loguniform64to32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|1.235558|1.123110|1.100122|+10.012%|0|True|
|250|decode|bf16-d256-h8-hk2-b1-paged256-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.190477|0.134451|1.416698|+41.670%|0|True|
|251|decode|bf16-d128-h16-hk4-b8-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null}`|0.188077|0.127539|1.474659|+47.466%|0|True|
|252|decode|bf16-d128-h16-hk4-b32-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.268531|0.241850|1.110323|+11.032%|0|True|
|253|decode|bf16-d256-h8-hk2-b32-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.422349|0.174125|2.425552|+142.555%|0|True|
|254|decode|bf16-d256-h8-hk2-b1-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.182323|0.126336|1.443161|+44.316%|0|True|
|255|decode|bf16-d256-h8-hk2-b1-paged256-uniform4096|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.203603|0.191654|1.062346|+6.235%|0|True|
|256|decode|bf16-d128-h16-hk4-b128-paged256-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.263136|0.247629|1.062623|+6.262%|0|True|
|257|decode|bf16-d256-h8-hk2-b8-paged256-uniform32768|kad-direct|`{"num_splits":2,"pack_gqa":null}`|0.517850|0.488403|1.060291|+6.029%|0|True|
|258|decode|bf16-d64-h32-hk8-b8-dense-loguniform64to32768|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.366170|0.271770|1.347353|+34.735%|0|True|
|259|decode|bf16-d128-h16-hk4-b8-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.253261|0.190458|1.329749|+32.975%|0|True|
|260|decode|bf16-d128-h16-hk4-b8-paged256-uniform4096|kad-direct|`{"num_splits":4,"pack_gqa":null,"tile_m":16,"tile_n":64}`|0.251776|0.165574|1.520622|+52.062%|0|True|
|261|decode|bf16-d64-h32-hk8-b8-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":16,"tile_n":64}`|0.193306|0.128890|1.499777|+49.978%|0|True|
|262|decode|bf16-d64-h32-hk8-b32-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.259642|0.236122|1.099610|+9.961%|0|True|
|263|decode|bf16-d64-h32-hk8-b1-paged256-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.267597|0.184858|1.447583|+44.758%|0|True|
|264|decode|bf16-d256-h8-hk2-b32-dense-one_long_short128|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.252307|0.237120|1.064049|+6.405%|0|True|
|265|decode|bf16-d256-h8-hk2-b128-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.684429|0.194726|3.514823|+251.482%|0|True|
|266|decode|bf16-d64-h32-hk8-b32-dense-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.395277|0.349754|1.130158|+13.016%|0|True|
|267|decode|bf16-d128-h16-hk4-b32-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":128,"tile_n":64}`|1.636102|1.537434|1.064178|+6.418%|0|True|
|268|decode|bf16-d256-h8-hk2-b1-dense-uniform4096|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.186349|0.172832|1.078208|+7.821%|0|True|
|269|decode|bf16-d128-h16-hk4-b1-paged256-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.249786|0.185510|1.346478|+34.648%|0|True|
|270|decode|bf16-d64-h32-hk8-b1-dense-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.181773|0.121382|1.497522|+49.752%|0|True|
|271|decode|bf16-d128-h16-hk4-b32-dense-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.386458|0.291552|1.325519|+32.552%|0|True|
|272|decode|bf16-d256-h8-hk2-b1-paged256-one_long_short128|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.203149|0.168928|1.202576|+20.258%|0|True|
|273|decode|bf16-d128-h16-hk4-b128-paged256-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|5.812429|5.706470|1.018568|+1.857%|0|True|
|274|decode|bf16-d256-h8-hk2-b8-dense-uniform4096|kad-direct|`{"num_splits":16,"pack_gqa":null}`|0.241178|0.156877|1.537370|+53.737%|0|True|
|275|decode|bf16-d128-h16-hk4-b1-dense-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.185158|0.123277|1.501973|+50.197%|0|True|
|276|decode|bf16-d128-h16-hk4-b1-paged256-loguniform64to32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.253491|0.133779|1.894848|+89.485%|0|True|
|277|decode|bf16-d256-h8-hk2-b128-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":32}`|0.845184|0.818643|1.032420|+3.242%|0|True|
|278|decode|bf16-d256-h8-hk2-b8-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":16,"tile_n":64}`|0.203693|0.133670|1.523844|+52.384%|0|True|
|279|decode|bf16-d128-h16-hk4-b128-paged256-uniform4096|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":96}`|0.918976|0.821875|1.118145|+11.815%|0|True|
|280|decode|bf16-d256-h8-hk2-b1-dense-uniform32768|kad-direct|`{"num_splits":0,"pack_gqa":null}`|0.183718|0.174803|1.051001|+5.100%|0|True|
|281|decode|bf16-d128-h16-hk4-b8-paged256-uniform32768|kad-direct|`{"num_splits":2,"pack_gqa":null}`|0.603821|0.488314|1.236543|+23.654%|0|True|
|282|decode|bf16-d64-h32-hk8-b32-paged256-loguniform64to32768|kad-direct|`{"num_splits":2,"pack_gqa":null,"tile_m":64,"tile_n":128}`|0.388179|0.361056|1.075122|+7.512%|0|True|
|283|decode|bf16-d256-h8-hk2-b32-paged256-uniform512|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|0.190208|0.139315|1.365307|+36.531%|0|True|
|284|decode|bf16-d64-h32-hk8-b8-paged256-loguniform64to32768|kad-direct|`{"num_splits":8,"pack_gqa":null}`|0.351315|0.271149|1.295655|+29.565%|0|True|
|285|decode|bf16-d256-h8-hk2-b32-dense-uniform32768|kad-direct|`{"num_splits":1,"pack_gqa":null,"tile_m":64,"tile_n":64}`|1.611885|1.517472|1.062217|+6.222%|0|True|

## 汇总

- Forward: 165 个，几何平均 1.015981642x（+1.598%），KDA direct 35，PR fallback 130
- Decode: 120 个，几何平均 1.251857565x（+25.186%），KDA direct 120，PR fallback 0
- 合计: 285 个，几何平均 1.109333954x（+10.933%），KDA direct 155，PR fallback 130
