-----create data-----

# create_circle_orbit.py
円軌道を離散な点で作成

# sin_3d_const.py
sinの表面をメッシュから等間隔で作成
凹凸判定の判定データとして利用

# sin_3d_known.py
sinの表面からデータ点をランダムに作成（不確実性を含まない）

# sin_3d_unknown.py
sinの表面からデータ点をランダムに作成（不確実な領域を含む）

-----search-----

# greedy_step_normal.py
法線方向を考慮した貪欲探索
クラス化
GPIS、VREP

# greedy_step_not_normal.py
法線方向を考慮せず、z方向のみ移動する貪欲探索
クラス化
GPIS、VREP

-----show result-----

# orbit_plot.py
探索後、ロボットが通った軌道を表示

# judge_shape_1.py
sin_3d_const.pyで作成したデータを用いて、
GPISで推定した形状のヘッセ行列から固有値を求める
npy形式で固有値を保存

# judge_shape_2.py
judge_shape_1.pyで保存した固有値を用いて、
凹凸を判定し、制度を計算
3D表示

# result_greedy.ipynb
データ点、凹凸判定結果、実際の軌道結果をすべて表示