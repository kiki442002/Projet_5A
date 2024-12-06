#define N_MELS 30

const float32_t mel_filters_non_zero_data[] = {
    2.5215744e-05, 5.043149e-05, 7.564723e-05, 0.00010086298, 0.00012607872, 0.00015129446, 0.00013754467, 0.00011232893, 8.7113185e-05, 6.189744e-05, 3.6681697e-05, 1.1465951e-05, 1.948277e-05, 4.469851e-05, 6.991426e-05, 9.513e-05, 0.00012034574, 0.00014556148, 0.00014327765, 0.00011806191, 9.284616e-05, 6.763041e-05, 4.241467e-05, 1.7198925e-05, 1.3749795e-05, 3.896554e-05, 6.418129e-05, 8.939702e-05, 0.00011461277, 0.00013982852, 0.00014901062, 0.00012379487, 9.8579134e-05, 7.336339e-05, 4.814765e-05, 2.2931901e-05, 7.988604e-06, 3.31156e-05, 5.82426e-05, 8.33696e-05, 0.0001084966, 0.0001336236, 0.00015419898, 0.00012907198, 0.000103944985, 7.881798e-05, 5.3690987e-05, 2.8563989e-05, 3.4369925e-06, 2.278515e-06, 2.7435426e-05, 5.259234e-05, 7.7749246e-05, 0.00010290616, 0.00012806307, 0.00015321998, 0.00013494524, 0.00010978833, 8.4631414e-05, 5.9474503e-05, 3.4317596e-05, 9.1606835e-06, 2.1766613e-05, 4.6982357e-05, 7.21981e-05, 9.7413846e-05, 0.00012262959, 0.00014784533, 0.0001409938, 0.00011577806, 9.056232e-05, 6.534657e-05, 4.0130828e-05, 1.4915082e-05, 1.6033639e-05, 4.124938e-05, 6.646512e-05, 9.168087e-05, 0.00011689661, 0.00014211236, 0.00014672679, 0.00012151104, 9.629529e-05, 7.107954e-05, 4.5863802e-05, 2.0648056e-05, 1.0288385e-05, 3.5474077e-05, 6.065977e-05, 8.584546e-05, 0.00011103115, 0.00013621684, 0.00015227805, 0.00012709237, 0.00010190666, 7.672097e-05, 5.1535284e-05, 2.6349591e-05, 1.1638991e-06, 4.546422e-06, 2.9644774e-05, 5.4743126e-05, 7.9841484e-05, 0.000104939834, 0.00013003818, 0.00015513653, 0.0001323579, 0.00010725955, 8.21612e-05, 5.7062847e-05, 3.19645e-05, 6.866148e-06, 2.3488172e-05, 4.8114387e-05, 7.27406e-05, 9.736682e-05, 0.000121993035, 0.00014661925, 0.00013632375, 0.000112876936, 8.943011e-05, 6.5983295e-05, 4.2536474e-05, 1.9089654e-05, 1.5718593e-05, 3.735672e-05, 5.8994854e-05, 8.0632984e-05, 0.000102271115, 0.00012390925, 0.00013790549, 0.00011842102, 9.8936565e-05, 7.9452104e-05, 5.9967646e-05, 4.0483184e-05, 2.0998725e-05, 1.5142648e-06, 3.2642865e-06, 2.0830088e-05, 3.839589e-05, 5.5961693e-05, 7.3527495e-05, 9.1093294e-05, 0.0001086591, 0.0001262249, 0.00011293617, 9.704747e-05, 8.1158774e-05, 6.527007e-05, 4.9381368e-05, 3.3492666e-05, 1.7603965e-05, 1.7152647e-06, 1.3276647e-05, 2.7672055e-05, 4.2067466e-05, 5.646287e-05, 7.0858274e-05, 8.525368e-05, 9.964909e-05, 0.0001140445, 0.00010398324, 9.096224e-05, 7.7941244e-05, 6.492024e-05, 5.189924e-05, 3.8878236e-05, 2.5857235e-05, 1.28362335e-05, 1.0485156e-05, 2.2239223e-05, 3.399329e-05, 4.5747354e-05, 5.7501424e-05, 6.925549e-05, 8.100955e-05, 9.276362e-05, 0.00010420003, 9.356819e-05, 8.293634e-05, 7.230451e-05, 6.167266e-05, 5.1040817e-05, 4.0408977e-05, 2.9777135e-05, 1.9145291e-05, 8.513448e-06, 1.3669485e-07, 9.769874e-06, 1.9403053e-05, 2.9036231e-05, 3.8669412e-05, 4.8302587e-05, 5.7935766e-05, 6.7568944e-05, 7.720212e-05, 8.683531e-05, 9.281292e-05, 8.409947e-05, 7.538603e-05, 6.6672575e-05, 5.7959132e-05, 4.9245682e-05, 4.0532235e-05, 3.1818785e-05, 2.3105338e-05, 1.4391891e-05, 5.6784443e-06, 1.57016e-06, 9.450514e-06, 1.7330867e-05, 2.5211222e-05, 3.3091575e-05, 4.097193e-05, 4.8852286e-05, 5.6732635e-05, 6.461299e-05, 7.249335e-05, 8.03737e-05, 8.302646e-05, 7.589849e-05, 6.877052e-05, 6.164254e-05, 5.4514567e-05, 4.7386595e-05, 4.025862e-05, 3.3130644e-05, 2.6002672e-05, 1.8874696e-05, 1.1746722e-05, 4.618748e-06, 2.2461097e-06, 8.694657e-06, 1.5143203e-05, 2.159175e-05, 2.8040296e-05, 3.4488843e-05, 4.093739e-05, 4.7385936e-05, 5.3834483e-05, 6.028303e-05, 6.6731576e-05, 7.318012e-05, 7.53053e-05, 6.947243e-05, 6.363956e-05, 5.7806694e-05, 5.1973824e-05, 4.6140955e-05, 4.0308085e-05, 3.4475215e-05, 2.8642347e-05, 2.2809478e-05, 1.6976608e-05, 1.1143741e-05, 5.3108706e-06, 1.8549821e-06, 7.124441e-06, 1.23939e-05, 1.7663358e-05, 2.2932816e-05, 2.8202276e-05, 3.3471733e-05, 3.8741193e-05, 4.4010652e-05, 4.9280112e-05, 5.4549568e-05, 5.981903e-05, 6.508849e-05, 6.945982e-05, 6.4693464e-05, 5.9927104e-05, 5.516075e-05, 5.0394396e-05, 4.562804e-05, 4.0861683e-05, 3.609533e-05, 3.1328975e-05, 2.6562619e-05, 2.1796264e-05, 1.7029908e-05, 1.2263554e-05, 7.497198e-06, 2.7308429e-06, 3.8633132e-07, 4.703244e-06, 9.0201565e-06, 1.3337069e-05, 1.7653982e-05, 2.1970893e-05, 2.6287806e-05, 3.060472e-05, 3.492163e-05, 3.9238545e-05, 4.3555454e-05, 4.7872367e-05, 5.218928e-05, 5.6506193e-05, 6.0823102e-05, 6.162889e-05, 5.772413e-05, 5.3819378e-05, 4.9914626e-05, 4.600987e-05, 4.210512e-05, 3.8200364e-05, 3.429561e-05, 3.0390856e-05, 2.6486103e-05, 2.258135e-05, 1.8676596e-05, 1.4771842e-05, 1.0867089e-05, 6.962335e-06, 3.0575816e-06, 1.5072796e-06, 5.0367253e-06, 8.566171e-06, 1.2095617e-05, 1.5625063e-05, 1.9154508e-05, 2.2683953e-05, 2.62134e-05, 2.9742843e-05, 3.3272292e-05, 3.6801735e-05, 4.0331182e-05, 4.386063e-05, 4.7390073e-05, 5.091952e-05, 5.4448963e-05, 5.6520028e-05, 5.332756e-05, 5.0135088e-05, 4.694262e-05, 4.3750148e-05, 4.055768e-05, 3.7365207e-05, 3.4172735e-05, 3.0980264e-05, 2.7787795e-05, 2.4595323e-05, 2.1402853e-05, 1.8210383e-05, 1.5017913e-05, 1.1825442e-05, 8.632971e-06, 5.440501e-06, 2.2480303e-06, 6.26631e-07, 3.5148753e-06, 6.403119e-06, 9.291364e-06, 1.2179608e-05, 1.50678525e-05, 1.7956096e-05, 2.084434e-05, 2.3732586e-05, 2.662083e-05, 2.9509072e-05, 3.2397318e-05, 3.528556e-05, 3.8173806e-05, 4.106205e-05, 4.3950295e-05, 4.6838537e-05, 4.9726783e-05, 5.0987725e-05, 4.8375237e-05, 4.5762747e-05, 4.3150263e-05, 4.0537772e-05, 3.7925285e-05, 3.5312798e-05, 3.2700307e-05, 3.0087822e-05, 2.7475333e-05, 2.4862846e-05, 2.2250357e-05, 1.9637868e-05, 1.702538e-05, 1.44128935e-05, 1.18004045e-05, 9.1879165e-06, 6.575429e-06, 3.962941e-06, 1.3504527e-06, 6.991111e-07, 3.0623012e-06, 5.425491e-06, 7.788682e-06, 1.0151872e-05, 1.2515061e-05, 1.4878252e-05, 1.7241442e-05, 1.9604631e-05, 2.1967822e-05, 2.4331013e-05, 2.66942e-05, 2.9057392e-05, 3.1420583e-05, 3.378377e-05, 3.614696e-05, 3.8510152e-05, 4.0873343e-05, 4.3236534e-05, 4.559972e-05, 4.57887e-05, 4.3651136e-05, 4.151357e-05, 3.9376006e-05, 3.7238442e-05, 3.510088e-05, 3.296332e-05, 3.0825755e-05, 2.868819e-05, 2.6550624e-05, 2.441306e-05, 2.2275499e-05, 2.0137935e-05, 1.8000372e-05, 1.5862806e-05, 1.3725244e-05, 1.158768e-05, 9.4501165e-06, 7.312553e-06, 5.1749894e-06, 3.0374256e-06, 8.99862e-07, 9.3397307e-07, 2.8673528e-06, 4.8007323e-06, 6.7341125e-06, 8.667492e-06, 1.0600872e-05, 1.2534251e-05, 1.4467632e-05, 1.640101e-05, 1.8334391e-05, 2.0267771e-05, 2.2201151e-05, 2.413453e-05, 2.606791e-05, 2.800129e-05, 2.9934672e-05, 3.186805e-05, 3.380143e-05, 3.573481e-05, 3.766819e-05, 3.9601568e-05, 4.1534946e-05, 4.1336265e-05, 3.9587474e-05, 3.7838683e-05, 3.6089896e-05, 3.4341105e-05, 3.2592314e-05, 3.0843526e-05, 2.9094736e-05, 2.7345945e-05, 2.5597155e-05, 2.3848366e-05, 2.2099577e-05, 2.0350788e-05, 1.8601997e-05, 1.6853208e-05, 1.5104418e-05, 1.3355629e-05, 1.1606838e-05, 9.8580485e-06, 8.109259e-06, 6.3604693e-06, 4.61168e-06, 2.8628904e-06, 1.1141008e-06, 9.1589305e-07, 2.4976798e-06, 4.079466e-06, 5.6612525e-06, 7.2430394e-06, 8.824826e-06, 1.0406613e-05, 1.1988399e-05, 1.3570186e-05, 1.5151972e-05, 1.6733758e-05, 1.8315546e-05, 1.9897332e-05, 2.147912e-05, 2.3060904e-05, 2.4642692e-05, 2.622448e-05, 2.7806263e-05, 2.9388051e-05, 3.0969837e-05, 3.2551623e-05, 3.4133413e-05, 3.5715195e-05, 3.7296984e-05, 3.7785423e-05, 3.635466e-05, 3.4923894e-05, 3.3493132e-05, 3.2062366e-05, 3.06316e-05, 2.9200834e-05, 2.777007e-05, 2.6339305e-05, 2.490854e-05, 2.3477774e-05, 2.204701e-05, 2.0616246e-05, 1.9185482e-05, 1.7754715e-05, 1.6323951e-05, 1.4893186e-05, 1.34624215e-05, 1.2031655e-05, 1.0600891e-05, 9.170126e-06, 7.7393615e-06, 6.3085963e-06, 4.8778315e-06, 3.4470665e-06, 2.0163018e-06, 5.855369e-07, 4.6976837e-07, 1.7641447e-06, 3.0585209e-06, 4.352897e-06, 5.647273e-06, 6.941649e-06, 8.236026e-06, 9.530402e-06, 1.0824779e-05, 1.2119154e-05, 1.3413531e-05, 1.4707906e-05, 1.6002283e-05, 1.729666e-05, 1.8591034e-05, 1.988541e-05, 2.1179789e-05, 2.2474165e-05, 2.376854e-05, 2.5062916e-05, 2.6357293e-05, 2.765167e-05, 2.8946044e-05, 3.024042e-05, 3.1534797e-05, 3.282917e-05, 3.412355e-05, 3.396162e-05, 3.2790824e-05, 3.162003e-05, 3.0449235e-05, 2.9278439e-05, 2.8107643e-05, 2.6936848e-05, 2.5766052e-05, 2.459526e-05, 2.3424464e-05, 2.225367e-05, 2.1082873e-05, 1.9912079e-05, 1.8741282e-05, 1.7570488e-05, 1.6399692e-05, 1.5228898e-05, 1.4058102e-05, 1.2887307e-05, 1.1716512e-05, 1.0545717e-05, 9.374921e-06, 8.204126e-06, 7.033331e-06, 5.862536e-06, 4.6917407e-06, 3.5209457e-06, 2.3501507e-06, 1.1793554e-06, 8.560246e-09, 6.2538453e-07, 1.6840081e-06, 2.7426318e-06, 3.8012552e-06, 4.8598786e-06, 5.918502e-06, 6.9771263e-06, 8.0357495e-06, 9.094373e-06, 1.0152997e-05, 1.121162e-05, 1.2270243e-05, 1.3328867e-05, 1.438749e-05, 1.5446114e-05, 1.6504739e-05, 1.7563361e-05, 1.8621986e-05, 1.9680609e-05, 2.0739231e-05, 2.1797854e-05, 2.2856479e-05, 2.3915103e-05, 2.4973726e-05, 2.603235e-05, 2.7090973e-05, 2.8149598e-05, 2.920822e-05, 3.0266845e-05, 3.1325468e-05, 3.0382658e-05, 2.9425106e-05, 2.8467555e-05, 2.7510005e-05, 2.6552454e-05, 2.5594902e-05, 2.4637351e-05, 2.36798e-05, 2.2722248e-05, 2.1764698e-05, 2.0807147e-05, 1.9849596e-05, 1.8892046e-05, 1.7934493e-05, 1.6976943e-05, 1.6019392e-05, 1.5061841e-05, 1.410429e-05, 1.3146739e-05, 1.2189187e-05, 1.1231637e-05, 1.0274085e-05, 9.316534e-06, 8.358983e-06, 7.401432e-06, 6.443881e-06, 5.48633e-06, 4.528779e-06, 3.5712276e-06, 2.6136765e-06, 1.6561255e-06, 6.9857447e-07, 8.600964e-07, 1.7265277e-06, 2.5929592e-06, 3.4593904e-06, 4.3258215e-06, 5.192253e-06, 6.0586844e-06, 6.925116e-06, 7.7915465e-06, 8.657978e-06, 9.5244095e-06, 1.03908405e-05, 1.1257272e-05, 1.2123703e-05, 1.2990135e-05, 1.3856566e-05, 1.4722998e-05, 1.5589429e-05, 1.645586e-05, 1.732229e-05, 1.8188723e-05, 1.9055153e-05, 1.9921585e-05, 2.0788017e-05, 2.1654449e-05, 2.2520879e-05, 2.338731e-05, 2.425374e-05, 2.5120173e-05, 2.5986603e-05, 2.6853037e-05, 2.7719467e-05, 2.8139606e-05, 2.7355896e-05, 2.657219e-05, 2.578848e-05, 2.5004772e-05, 2.4221064e-05, 2.3437355e-05, 2.2653647e-05, 2.1869939e-05, 2.1086229e-05, 2.0302521e-05, 1.9518811e-05, 1.8735103e-05, 1.7951397e-05, 1.7167687e-05, 1.638398e-05, 1.5600272e-05, 1.4816562e-05, 1.4032854e-05, 1.3249145e-05, 1.2465437e-05, 1.1681729e-05, 1.089802e-05, 1.0114311e-05, 9.330603e-06, 8.546895e-06, 7.763187e-06, 6.979478e-06, 6.1957694e-06, 5.4120605e-06, 4.6283526e-06, 3.8446437e-06, 3.0609356e-06, 2.2772272e-06, 1.4935185e-06, 7.098101e-07, 1.9168986e-07, 9.004521e-07, 1.6092142e-06, 2.3179764e-06, 3.0267388e-06, 3.735501e-06, 4.444263e-06, 5.1530255e-06, 5.861788e-06, 6.5705503e-06, 7.279312e-06, 7.988075e-06, 8.696837e-06, 9.4055995e-06, 1.01143605e-05, 1.0823123e-05, 1.1531885e-05, 1.2240647e-05, 1.294941e-05, 1.3658171e-05, 1.4366935e-05, 1.5075696e-05, 1.5784459e-05, 1.649322e-05, 1.7201983e-05, 1.7910746e-05, 1.8619508e-05, 1.9328269e-05, 2.0037032e-05, 2.0745794e-05, 2.1454558e-05, 2.2163318e-05, 2.287208e-05, 2.3580844e-05, 2.4289606e-05, 2.4998368e-05, 2.5579848e-05, 2.4938756e-05, 2.4297662e-05, 2.3656568e-05, 2.3015476e-05, 2.2374385e-05, 2.173329e-05, 2.1092197e-05, 2.0451105e-05, 1.9810012e-05, 1.9168918e-05, 1.8527826e-05, 1.7886734e-05, 1.724564e-05, 1.6604547e-05, 1.5963455e-05, 1.5322363e-05, 1.4681268e-05, 1.4040176e-05, 1.3399083e-05, 1.2757991e-05, 1.2116897e-05, 1.1475805e-05, 1.0834711e-05, 1.0193619e-05, 9.552526e-06, 8.911433e-06, 8.27034e-06, 7.6292467e-06, 6.9881544e-06, 6.347061e-06, 5.7059683e-06, 5.0648755e-06, 4.4237827e-06, 3.7826896e-06, 3.1415968e-06, 2.500504e-06, 1.859411e-06, 1.2183182e-06, 5.7722536e-07, 5.4681276e-08, 6.345886e-07, 1.2144959e-06, 1.7944031e-06, 2.3743107e-06, 2.9542182e-06, 3.5341254e-06, 4.1140324e-06, 4.69394e-06, 5.2738474e-06, 5.8537544e-06, 6.4336623e-06, 7.013569e-06, 7.5934768e-06, 8.173384e-06, 8.753291e-06, 9.333199e-06, 9.913106e-06, 1.0493013e-05, 1.1072921e-05, 1.1652829e-05, 1.2232736e-05, 1.2812642e-05, 1.339255e-05, 1.3972458e-05, 1.45523645e-05, 1.5132272e-05, 1.571218e-05, 1.6292086e-05, 1.6871994e-05, 1.7451903e-05, 1.803181e-05, 1.8611716e-05, 1.9191622e-05, 1.977153e-05, 2.0351437e-05, 2.0931346e-05, 2.1511254e-05, 2.209116e-05, 2.2671067e-05, 2.3140947e-05, 2.2616407e-05, 2.2091865e-05, 2.1567324e-05, 2.1042784e-05, 2.0518244e-05, 1.9993702e-05, 1.9469164e-05, 1.8944622e-05, 1.8420082e-05, 1.7895542e-05, 1.7371001e-05, 1.6846461e-05, 1.6321921e-05, 1.579738e-05, 1.5272839e-05, 1.47483e-05, 1.4223759e-05, 1.3699218e-05, 1.3174678e-05, 1.2650137e-05, 1.2125596e-05, 1.1601055e-05, 1.1076516e-05, 1.0551975e-05, 1.0027435e-05, 9.502894e-06, 8.978353e-06, 8.453813e-06, 7.929272e-06, 7.4047325e-06, 6.8801924e-06, 6.3556513e-06, 5.831111e-06, 5.3065705e-06, 4.78203e-06, 4.2574898e-06, 3.7329492e-06, 3.2084085e-06, 2.6838682e-06, 2.1593278e-06, 1.6347873e-06, 1.1102468e-06, 5.8570635e-07, 6.116587e-08, 4.726841e-08, 5.217415e-07, 9.962147e-07, 1.4706877e-06, 1.9451609e-06, 2.419634e-06, 2.8941072e-06, 3.3685803e-06, 3.8430535e-06, 4.317527e-06, 4.792e-06, 5.2664727e-06, 5.7409457e-06, 6.2154195e-06, 6.6898924e-06, 7.164366e-06, 7.638839e-06, 8.113311e-06, 8.5877855e-06, 9.062258e-06, 9.536731e-06, 1.0011205e-05, 1.0485677e-05, 1.0960151e-05, 1.1434624e-05, 1.1909097e-05, 1.238357e-05, 1.2858043e-05, 1.3332517e-05, 1.3806989e-05, 1.4281462e-05, 1.4755936e-05, 1.5230408e-05, 1.5704882e-05, 1.6179356e-05, 1.6653828e-05, 1.71283e-05, 1.7602775e-05, 1.8077248e-05, 1.855172e-05, 1.9026194e-05, 1.9500667e-05, 1.997514e-05, 2.0449614e-05, 2.0924086e-05, 2.0600286e-05, 2.0171114e-05, 1.9741941e-05, 1.9312769e-05, 1.8883595e-05, 1.8454424e-05, 1.8025252e-05, 1.7596078e-05, 1.7166905e-05, 1.6737733e-05, 1.630856e-05, 1.5879388e-05, 1.5450214e-05, 1.5021042e-05, 1.4591869e-05, 1.4162696e-05, 1.3733524e-05, 1.3304352e-05, 1.287518e-05, 1.2446007e-05, 1.2016834e-05, 1.1587661e-05, 1.1158489e-05, 1.0729316e-05, 1.0300143e-05, 9.870971e-06, 9.441797e-06, 9.012626e-06, 8.583453e-06, 8.15428e-06, 7.725107e-06, 7.2959347e-06, 6.866762e-06, 6.43759e-06, 6.008417e-06, 5.5792443e-06, 5.1500715e-06, 4.7208987e-06, 4.2917263e-06, 3.8625535e-06, 3.433381e-06, 3.0042086e-06, 2.5750358e-06, 2.1458632e-06, 1.7166905e-06, 1.2875179e-06, 8.5834523e-07, 4.2917262e-07};

const int16_t mel_filters_zeros_before[] = {
    1, 7, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 77, 85, 93, 103, 114, 126, 139, 154, 170, 188, 208, 230, 254, 281, 311, 343, 379, 419};

const uint8_t mel_filters_num_non_zero[] = {
    12, 12, 12, 13, 13, 12, 12, 13, 13, 12, 14, 16, 16, 18, 21, 23, 25, 28, 31, 34, 38, 42, 46, 51, 57, 62, 68, 76, 85, 93};
