14
15 -try:
16 -    import matplotlib.pyplot as plt
17 -    HAVE_PLOT = True
18 -except Exception:
19 -    HAVE_PLOT = False
15 +# Optional plotting (opt-in to avoid numpy/matplotlib build issues)
16 +HAVE_PLOT = False
17 +if os.environ.get("PLOT", "").lower() in ("1", "true", "yes"):
18 +    try:
19 +        import matplotlib.pyplot as plt
20 +        HAVE_PLOT = True
21 +    except Exception:
22 +        HAVE_PLOT = False
23

