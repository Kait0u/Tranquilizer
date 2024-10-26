# Functions
function Write-Separator {
    Write-Host ("-" * 60)
}

# -----------------------------------------------------------------------------------------------------------------
# Actual script
# -----------------------------------------------------------------------------------------------------------------

# Define the data directory and batch size
$dataDir = ".\datasets\pexels-110k-512p-min-jpg\selected_images"
$batchSize = 1
$dataLimit = 80
$epochs = 160
$cpEvery = 20

# Boolean values

$command1 = "python .\tranq_train.py --data_dir $dataDir --epochs $epochs --batch_size $batchSize --checkpoint_every $cpEvery --data_limit $dataLimit"
$command2 = "python .\tranq_train.py --data_dir $dataDir --epochs $epochs --batch_size $batchSize --checkpoint_every $cpEvery --data_limit $dataLimit --grayscale"

$commands = @($command1, $command2)

foreach ($comm in $commands) {
    Write-Separator
    Write-Host "Executing: $comm"
    Write-Separator

    Invoke-Expression $comm
}
