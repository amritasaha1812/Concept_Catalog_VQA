grep 'Starting cluster\|Epoch[ ]*[0-9]*[ ]*Loss\|Number of classes ==  1, hence skipping this cluster' out/o*.txt | cut -f2- -d: | awk 'BEGIN{FS=" "; max=0.0; line=""} {if($1=="Starting"){ print line; print $0; max=0.0} else{if (max<$NF){ max=$NF;line = $0}}} END { print line}'| sed 's/Score/Score[/g' | cut -f2 -d'[' | sed 's/]\n/\t/g' | sed 's/]//g'
