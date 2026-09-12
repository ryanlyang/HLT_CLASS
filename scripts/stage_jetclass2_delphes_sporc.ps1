# Stage ONLY the Delphes migration/readiness package. Never commits or pushes.
$ErrorActionPreference = 'Stop'
$Jc2Repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $Jc2Repo
try {
    $Jc2ExistingIndex = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Git index' }
    if ($Jc2ExistingIndex.Count -ne 0) {
        throw 'Git index is not empty. Inspect existing staged work before staging this migration.'
    }
    $Jc2Paths = @(
        Get-ChildItem -LiteralPath 'src/hlt_classification/jetclass2_delphes' -File -Filter '*.py'
        Get-ChildItem -LiteralPath 'scripts' -File -Filter '*jetclass2_delphes*.py'
        Get-ChildItem -LiteralPath 'sbatch' -File -Filter '*jetclass2_delphes*.sh'
        Get-ChildItem -LiteralPath 'tests' -File -Filter 'test_jetclass2_delphes*.py'
        Get-Item -LiteralPath 'scripts/stage_jetclass2_delphes_sporc.ps1',
            'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md',
            'docs/contracts/JETCLASS2_DELPHES.md',
            'docs/plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md',
            'docs/JETCLASS2_DELPHES_SPORC_READINESS.md', 'pyproject.toml'
    ) | ForEach-Object { $_.FullName }
    git add -- @Jc2Paths
    if ($LASTEXITCODE -ne 0) { throw 'Scoped staging failed; inspect the index' }
    # HANDOFF also contains an unrelated, unpublished salience section. Publish
    # only the Delphes additions; keep the complete working file untouched.
    $Jc2Handoff = [IO.File]::ReadAllText((Join-Path $Jc2Repo 'docs/HANDOFF.md')).Replace("`r`n", "`n")
    $Jc2SectionStart = $Jc2Handoff.IndexOf("## 2026-09-11: full-cardinality salience-matching implementation`n")
    $Jc2SectionEnd = $Jc2Handoff.IndexOf("## 2026-09-02: Strategy-B adjacent learned-fusion handoff implementation`n")
    if ($Jc2SectionStart -lt 0 -or $Jc2SectionEnd -le $Jc2SectionStart) {
        throw 'HANDOFF boundaries changed; inspect and stage that mixed file manually'
    }
    $Jc2ScopedHandoff = $Jc2Handoff.Substring(0, $Jc2SectionStart) + $Jc2Handoff.Substring($Jc2SectionEnd)
    $Jc2GitStart = New-Object Diagnostics.ProcessStartInfo
    $Jc2GitStart.FileName = 'git'
    $Jc2GitStart.Arguments = 'hash-object -w --stdin'
    $Jc2GitStart.WorkingDirectory = $Jc2Repo
    $Jc2GitStart.UseShellExecute = $false
    $Jc2GitStart.CreateNoWindow = $true
    $Jc2GitStart.RedirectStandardInput = $true
    $Jc2GitStart.RedirectStandardOutput = $true
    $Jc2GitProcess = [Diagnostics.Process]::Start($Jc2GitStart)
    # BaseStream avoids Windows PowerShell 5's legacy pipeline encoding and
    # does not depend on the newer StandardInputEncoding .NET property.
    $Jc2HandoffBytes = [Text.Encoding]::UTF8.GetBytes($Jc2ScopedHandoff)
    $Jc2GitProcess.StandardInput.BaseStream.Write($Jc2HandoffBytes, 0, $Jc2HandoffBytes.Length)
    $Jc2GitProcess.StandardInput.BaseStream.Close()
    $Jc2HandoffHash = $Jc2GitProcess.StandardOutput.ReadToEnd().Trim()
    $Jc2GitProcess.WaitForExit()
    if ($Jc2GitProcess.ExitCode -ne 0 -or $Jc2HandoffHash -notmatch '^[a-f0-9]{40}$') {
        throw 'Could not construct the scoped HANDOFF index entry'
    }
    git update-index --cacheinfo "100644,$Jc2HandoffHash,docs/HANDOFF.md"
    if ($LASTEXITCODE -ne 0) { throw 'Could not stage the scoped HANDOFF index entry' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
    git diff --cached --stat
    Write-Host 'Only Delphes migration/readiness paths staged. No commit or push performed.'
    Write-Host 'Unrelated Scouting, salience (including HANDOFF), figures, worktrees, and README/index edits were left alone.'
} finally {
    Pop-Location
}
