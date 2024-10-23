# Functions
function Write-Separator {
    Write-Host ("-" * 60)
}

# -----------------------------------------------------------------------------------------------------------------
# Actual script
# -----------------------------------------------------------------------------------------------------------------

# Define the data directory and batch size
$dataDir = ".\datasets\pexels-110k-512p-min-jpg\images"
$batchSize = 32
$dataLimit = 400

# Define the epochs to iterate over
$epochs = @(1, 5, 10, 20, 50)

# Boolean values
$boolValues = @($false, $true)

# Loop through each epoch value and execute the command
foreach ($epochNum in $epochs) {
    foreach ($boolVal in $boolValues) {
        $command = "python .\tranq_train.py --data_dir $dataDir --epochs $epochNum --batch_size $batchSize --data_limit $dataLimit"
        if ($boolVal) {
            $command += " --grayscale"
        }

        Write-Separator
        Write-Host "Executing: $command"
        Write-Separator

        Invoke-Expression $command
    }
}
