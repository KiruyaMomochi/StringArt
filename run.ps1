#!/usr/bin/pwsh
$files = @(#"input/cat_denoised.png"
        #    "input/ada_denoised.png"
           "input/dog_denoised.png",
           "input/einstein_denoised.png")
$cmd = "build/src/StringArtCuda/StringArtCuda";

$pins = @(
    @(128, 512),
    @(256, 1024),
    @(384, 1536),
    @(512, 2048),
    @(640, 2560)
)

foreach ($file in $files)
{
    foreach ($pin in $pins)
    {
        Write-Host "Processing $file with $pin"
        & $cmd --input $file --pin-count $($pin[0]) --width $($pin[1])
    }
}
