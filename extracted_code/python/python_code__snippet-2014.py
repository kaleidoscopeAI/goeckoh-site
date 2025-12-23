"bash": """
    _pip_completion()
    {{
        COMPREPLY=( $( COMP_WORDS="${{COMP_WORDS[*]}}" \\
                       COMP_CWORD=$COMP_CWORD \\
                       PIP_AUTO_COMPLETE=1 $1 2>/dev/null ) )
    }}
    complete -o default -F _pip_completion {prog}
""",
"zsh": """
    #compdef -P pip[0-9.]#
    __pip() {{
      compadd $( COMP_WORDS="$words[*]" \\
                 COMP_CWORD=$((CURRENT-1)) \\
                 PIP_AUTO_COMPLETE=1 $words[1] 2>/dev/null )
    }}
    if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
      # autoload from fpath, call function directly
      __pip "$@"
    else
      # eval/source/. command, register function for later
      compdef __pip -P 'pip[0-9.]#'
    fi
""",
"fish": """
    function __fish_complete_pip
        set -lx COMP_WORDS (commandline -o) ""
        set -lx COMP_CWORD ( \\
            math (contains -i -- (commandline -t) $COMP_WORDS)-1 \\
        )
        set -lx PIP_AUTO_COMPLETE 1
        string split \\  -- (eval $COMP_WORDS[1])
    end
    complete -fa "(__fish_complete_pip)" -c {prog}
""",
"powershell": """
    if ((Test-Path Function:\\TabExpansion) -and -not `
        (Test-Path Function:\\_pip_completeBackup)) {{
        Rename-Item Function:\\TabExpansion _pip_completeBackup
    }}
    function TabExpansion($line, $lastWord) {{
        $lastBlock = [regex]::Split($line, '[|;]')[-1].TrimStart()
        if ($lastBlock.StartsWith("{prog} ")) {{
            $Env:COMP_WORDS=$lastBlock
            $Env:COMP_CWORD=$lastBlock.Split().Length - 1
            $Env:PIP_AUTO_COMPLETE=1
            (& {prog}).Split()
            Remove-Item Env:COMP_WORDS
            Remove-Item Env:COMP_CWORD
            Remove-Item Env:PIP_AUTO_COMPLETE
        }}
        elseif (Test-Path Function:\\_pip_completeBackup) {{
            # Fall back on existing tab expansion
            _pip_completeBackup $line $lastWord
        }}
    }}
""",
