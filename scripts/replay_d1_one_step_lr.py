#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_selected_intervention import main
if __name__=="__main__":
    if "--variant" not in sys.argv: sys.argv[1:1]=["--variant","d1-one-step"]
    main()
