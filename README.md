# Roblox Learning Enemy AI

Lua単体(Roblox標準機能のみ)で実装した、**プレイ中にリアルタイムで賢くなる敵NPC AI**です。
ニューラルネットワークとオンライン強化学習を外部ライブラリなしで自前実装し、プレイヤーの追跡と障害物回避を学習します。

## 特徴

- 外部ライブラリ・APIキー不要。Robloxの標準機能だけで完結
- 小型フィードフォワードニューラルネットワーク(6入力 → 8隠れ層 → 2出力)を自前実装
- 報酬ベースのオンライン学習(プレイ中に少しずつ重みが更新される軽量な強化学習)
- レイキャストによる壁検知で障害物回避を学習
- DataStoreで学習済みの重みを保存し、サーバー再起動をまたいで学習を引き継ぐ
- 死亡時の自動リスポーン、複数体の同時稼働に対応

## ファイル構成

| ファイル | 種別 | 役割 |
|---|---|---|
| `NeuralNetwork.lua` | ModuleScript | ニューラルネットワーク本体(順伝播・オンライン学習・シリアライズ) |
| `AIBrain.lua` | ModuleScript | NPC1体分の知覚(レイキャスト・プレイヤー検知)・行動決定・報酬計算 |
| `EnemyAIController.server.lua` | Script | NPCの生成・管理、メインループ、DataStore保存処理 |

## 仕組み

```
プレイヤーの位置・壁までの距離(レイキャスト)
        ↓ 入力ベクトル(6次元)
   NeuralNetwork (8隠れユニット)
        ↓ 出力(旋回方向・速度)
     NPCの移動
        ↓ 結果を評価
  報酬計算(接近+ / 離脱- / 壁衝突--)
        ↓
  オンライン重み更新(次の判断に反映)
```

毎フレーム「知覚 → 判断 → 行動 → 報酬計算 → 学習」のサイクルを回します。
本格的なディープラーニング(誤差逆伝播による多層学習)ではなく、Robloxの実行時間制約内で動作するよう設計した軽量な近似実装です。

## セットアップ

1. Roblox Studioでゲームを開く
2. `ServerScriptService` の中に以下を配置:
   - `NeuralNetwork.lua` → ModuleScriptとして配置(名前を `NeuralNetwork` に)
   - `AIBrain.lua` → ModuleScriptとして配置(名前を `AIBrain` に)
   - `EnemyAIController.server.lua` → Scriptとして配置
3. (任意) `Workspace` に `EnemySpawns` という Folder を作成し、中にPartを配置するとその位置からNPCがスポーンします。未設定の場合は原点付近に自動配置されます。
4. (任意) `Workspace` に `AIEnemy` という名前のNPCモデル(Humanoid + HumanoidRootPart必須)を用意すると、それがテンプレートとして複製されます。無い場合はコード側で簡易モデルが自動生成されます。
5. ゲームを実行(Play)すると敵NPCがスポーンし、学習を開始します。

## 設定値の調整

`AIBrain.lua` の冒頭にある定数で挙動を調整できます。

```lua
local LEARNING_RATE = 0.06        -- 学習率(大きいほど速く学習するが不安定になりやすい)
local REWARD_APPROACH = 1.0       -- プレイヤーに近づいたときの報酬
local REWARD_RETREAT = -1.0       -- プレイヤーから離れたときのペナルティ
local REWARD_WALL_HIT = -3.0      -- 壁衝突のペナルティ
local SENSE_RADIUS = 80           -- プレイヤーを感知する距離
```

`EnemyAIController.server.lua` ではNPCの数や保存間隔を調整できます。

```lua
local MAX_ENEMIES = 5
local SAVE_INTERVAL = 60 -- 秒
```

## 注意事項

- DataStoreを使用するため、Studio上でテストする場合は「Studioに対するAPIサービスへのアクセスを有効にする」設定をONにしてください(Game Settings → Security)。
- 学習は何百回ものインタラクションを経て徐々に効果が出てきます。数回プレイしただけでは大きな変化を感じにくい場合があります。
- 商用・大規模ゲームでの本格利用を想定したものではなく、学習目的・実験的な実装です。

## ライセンス

[MIT License](LICENSE)
