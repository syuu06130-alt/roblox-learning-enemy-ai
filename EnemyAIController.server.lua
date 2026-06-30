--[[
	EnemyAIController.server.lua
	-------------------------------------------------------------------
	NeuralNetwork.lua + AIBrain.lua を使い、実際にゲーム内で動く
	「学習する敵NPC」を生成・管理するメインスクリプト。

	設置方法:
	  - ServerScriptService 直下にこのスクリプトを置く
	  - 同じ階層(または ReplicatedStorage 等)に
	      ModuleScript "NeuralNetwork" (NeuralNetwork.lua の中身)
	      ModuleScript "AIBrain"       (AIBrain.lua の中身)
	    を配置し、下記 require のパスを実際の配置場所に合わせて調整する。
	  - Workspace に "EnemySpawns" という Folder を作り、中に Part を
	    いくつか置くと、その位置からNPCがスポーンする
	    (無ければ原点(0,5,0)に1体だけスポーンする)。

	機能:
	  - プレイヤーが近づくと最寄りの敵NPCがそのプレイヤーを追跡開始
	  - 各NPCはAIBrainを1つ持ち、フレームごとに学習しながら追跡・障害物回避
	  - 壁への衝突をTouchedイベントで検知し、AIBrainに罰として通知
	  - 一定間隔でDataStoreに重みを自動保存し、次回サーバー起動時に学習を引き継ぐ
--]]

local Players = game:GetService("Players")
local ServerScriptService = game:GetService("ServerScriptService")
local RunService = game:GetService("RunService")
local DataStoreService = game:GetService("DataStoreService")

-- ※配置場所に合わせてパスを調整してください
local NeuralNetwork = require(script.Parent:WaitForChild("NeuralNetwork"))
local AIBrain = require(script.Parent:WaitForChild("AIBrain"))

-- ============================================================
-- 設定
-- ============================================================

local MAX_ENEMIES = 5
local TARGET_SEARCH_RADIUS = 100 -- この距離内のプレイヤーを追跡対象にする
local SAVE_INTERVAL = 60 -- 秒。この間隔でDataStoreに重みを保存
local ENEMY_TEMPLATE_NAME = "AIEnemy" -- Workspace内に置く土台モデル名(無ければコードでPartを生成)

local AI_WEIGHTS_STORE = DataStoreService:GetDataStore("AIEnemyWeights_v1")

-- ============================================================
-- 内部状態
-- ============================================================

local activeEnemies = {} -- { [model] = { brain = AIBrain, humanoid = ..., rootPart = ... } }
local enemyIdCounter = 0

-- ============================================================
-- NPCモデル生成
-- ============================================================

local function createEnemyModel(spawnPosition, enemyId)
	-- Workspaceにあらかじめ用意したテンプレートがあればそれを複製、
	-- 無ければシンプルなブロック型NPCをコードで生成する
	local template = workspace:FindFirstChild(ENEMY_TEMPLATE_NAME)
	local model

	if template then
		model = template:Clone()
		model.Name = "AIEnemy_" .. enemyId
		model:SetPrimaryPartCFrame(CFrame.new(spawnPosition))
		model.Parent = workspace
	else
		-- フォールバック: コードでシンプルなNPCを組み立てる
		model = Instance.new("Model")
		model.Name = "AIEnemy_" .. enemyId

		local rootPart = Instance.new("Part")
		rootPart.Name = "HumanoidRootPart"
		rootPart.Size = Vector3.new(2, 2, 1)
		rootPart.Position = spawnPosition
		rootPart.BrickColor = BrickColor.new("Crimson")
		rootPart.TopSurface = Enum.SurfaceType.Smooth
		rootPart.BottomSurface = Enum.SurfaceType.Smooth
		rootPart.Parent = model

		local head = Instance.new("Part")
		head.Name = "Head"
		head.Shape = Enum.PartType.Ball
		head.Size = Vector3.new(1.4, 1.4, 1.4)
		head.Position = spawnPosition + Vector3.new(0, 1.7, 0)
		head.BrickColor = BrickColor.new("Crimson")
		head.Parent = model

		local neck = Instance.new("WeldConstraint")
		neck.Part0 = rootPart
		neck.Part1 = head
		neck.Parent = rootPart

		local humanoid = Instance.new("Humanoid")
		humanoid.WalkSpeed = 8
		humanoid.Parent = model

		model.PrimaryPart = rootPart
	end

	return model
end

-- ============================================================
-- DataStore: 重みの保存/読込
-- ============================================================

local function loadWeightsForSlot(slotIndex)
	local key = "enemy_slot_" .. tostring(slotIndex)
	local success, result = pcall(function()
		return AI_WEIGHTS_STORE:GetAsync(key)
	end)

	if success and result then
		return result
	end
	return nil
end

local function saveWeightsForSlot(slotIndex, weightsData)
	local key = "enemy_slot_" .. tostring(slotIndex)
	local success, err = pcall(function()
		AI_WEIGHTS_STORE:SetAsync(key, weightsData)
	end)
	if not success then
		warn("[EnemyAIController] 重み保存に失敗: " .. tostring(err))
	end
end

-- ============================================================
-- 壁衝突検知
-- ============================================================

local function setupWallCollisionDetection(rootPart, brain)
	rootPart.Touched:Connect(function(hit)
		-- プレイヤーキャラクターや他のNPC自身には反応しないようフィルタ
		if hit:IsDescendantOf(rootPart.Parent) then
			return
		end
		local hitModel = hit:FindFirstAncestorOfClass("Model")
		if hitModel and hitModel:FindFirstChildOfClass("Humanoid") then
			return -- 他のキャラクターとの接触は壁衝突として扱わない
		end
		brain:NotifyWallHit()
	end)
end

-- ============================================================
-- ターゲット(最も近いプレイヤー)の検索
-- ============================================================

local function findNearestPlayerTarget(position)
	local nearestPlayer = nil
	local nearestDist = TARGET_SEARCH_RADIUS

	for _, player in ipairs(Players:GetPlayers()) do
		local character = player.Character
		if character then
			local hrp = character:FindFirstChild("HumanoidRootPart")
			local hum = character:FindFirstChildOfClass("Humanoid")
			if hrp and hum and hum.Health > 0 then
				local dist = (hrp.Position - position).Magnitude
				if dist < nearestDist then
					nearestDist = dist
					nearestPlayer = hrp.Position
				end
			end
		end
	end

	return nearestPlayer
end

-- ============================================================
-- 敵NPCのスポーン
-- ============================================================

local function getSpawnPositions()
	local spawnsFolder = workspace:FindFirstChild("EnemySpawns")
	local positions = {}

	if spawnsFolder then
		for _, part in ipairs(spawnsFolder:GetChildren()) do
			if part:IsA("BasePart") then
				table.insert(positions, part.Position)
			end
		end
	end

	if #positions == 0 then
		-- フォールバック: 原点付近に分散スポーン
		for i = 1, MAX_ENEMIES do
			table.insert(positions, Vector3.new(i * 8, 5, 0))
		end
	end

	return positions
end

local function spawnEnemy(spawnPosition, slotIndex)
	enemyIdCounter = enemyIdCounter + 1
	local model = createEnemyModel(spawnPosition, enemyIdCounter)

	local humanoid = model:FindFirstChildOfClass("Humanoid")
	local rootPart = model:FindFirstChild("HumanoidRootPart") or model.PrimaryPart

	if not humanoid or not rootPart then
		warn("[EnemyAIController] NPCモデルの構成が不正です(HumanoidまたはHumanoidRootPartが無い)")
		model:Destroy()
		return
	end

	-- 保存済みの重みがあれば読み込み、無ければ新規ランダム個体
	local savedWeights = loadWeightsForSlot(slotIndex)
	local brain = AIBrain.new(rootPart, humanoid, savedWeights)

	setupWallCollisionDetection(rootPart, brain)

	activeEnemies[model] = {
		brain = brain,
		humanoid = humanoid,
		rootPart = rootPart,
		slotIndex = slotIndex,
	}

	model.Parent = workspace

	humanoid.Died:Connect(function()
		activeEnemies[model] = nil
		task.wait(5) -- リスポーン待機
		if model and model.Parent then
			model:Destroy()
		end
		spawnEnemy(spawnPosition, slotIndex)
	end)
end

local function initializeEnemies()
	local spawnPositions = getSpawnPositions()
	for i = 1, math.min(MAX_ENEMIES, #spawnPositions) do
		spawnEnemy(spawnPositions[i], i)
	end
end

-- ============================================================
-- メインループ: 全NPCのAIBrainを毎フレーム実行
-- ============================================================

RunService.Heartbeat:Connect(function(dt)
	for model, data in pairs(activeEnemies) do
		-- 既に破棄/死亡している場合はスキップ(Diedイベント側で後処理される)
		local isAlive = model.Parent ~= nil and data.humanoid.Health > 0

		if isAlive then
			local targetPosition = findNearestPlayerTarget(data.rootPart.Position)
			if targetPosition then
				data.brain:Step(targetPosition, dt)
			else
				-- ターゲットが見つからない場合はゆっくり停止
				data.humanoid:Move(Vector3.new(), false)
			end
		end
	end
end)

-- ============================================================
-- 定期的な重み自動保存(サーバーシャットダウン対策も兼ねる)
-- ============================================================

task.spawn(function()
	while true do
		task.wait(SAVE_INTERVAL)
		for model, data in pairs(activeEnemies) do
			if model.Parent then
				local weights = data.brain:GetWeights()
				saveWeightsForSlot(data.slotIndex, weights)
			end
		end
	end
end)

game:BindToClose(function()
	for model, data in pairs(activeEnemies) do
		local weights = data.brain:GetWeights()
		saveWeightsForSlot(data.slotIndex, weights)
	end
	-- BindToCloseはSetAsyncの完了をある程度待ってくれるが、
	-- 念のため短い待機を入れておく
	task.wait(1)
end)

-- ============================================================
-- 起動
-- ============================================================

initializeEnemies()

print("[EnemyAIController] 学習型敵AI " .. MAX_ENEMIES .. " 体を起動しました。")
