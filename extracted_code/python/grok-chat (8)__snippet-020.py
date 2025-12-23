    # Add a conceptual insight that links finance and healthcare (e.g., from a separate SN or pre-existing data)
    insight_synthesizer.ingest_super_node_insights(
        "EconomicHealth_SN",
        "Economic downturns correlate with increased mental health service demand.",
        ['economy', 'mental health', 'healthcare', 'correlation']
    )
    conscious_cube.add_conceptual_node("EconomicHealth_SN_Proxy", initial_strength=0.6)
    conscious_cube.update_node_activity("EconomicHealth_SN_Proxy", 0.7) # Simulate high activity for this link

    print("\n--- PHASE 3: Synthesis Tool Articulates Breakthroughs ---")
    # Synthesis Tool identifies clusters and articulates breakthroughs
    insight_synthesizer.identify_semantic_clusters()
    breakthrough_insights = insight_synthesizer.articulate_breakthroughs()
    if breakthrough_insights:
        for b_insight in breakthrough_insights:
            print(b_insight)
            # Conceptual: A breakthrough insight might strengthen a new connection in the Cube
            # Or even trigger the creation of a new conceptual node representing the breakthrough
            if "integrated risk models" in b_insight:
                conscious_cube.add_conceptual_node("IntegratedRiskModel_Insight", initial_strength=0.9)
                conscious_cube.graph.add_edge("IntegratedRiskModel_Insight", "Finance_SN_Proxy", weight=0.8)
                conscious_cube.graph.add_edge("IntegratedRiskModel_Insight", "SupplyChain_SN_Proxy", weight=0.7)
                conscious_cube._update_viz_properties("IntegratedRiskModel_Insight")
                conscious_cube._update_edge_viz_properties("IntegratedRiskModel_Insight", "Finance_SN_Proxy")
                conscious_cube._update_edge_viz_properties("IntegratedRiskModel_Insight", "SupplyChain_SN_Proxy")
            elif "healthcare delivery paradigms" in b_insight:
                conscious_cube.add_conceptual_node("NewHealthcareParadigm_Insight", initial_strength=0.9)
                conscious_cube.graph.add_edge("NewHealthcareParadigm_Insight", "Healthcare_SN_Proxy", weight=0.8)
                conscious_cube.graph.add_edge("NewHealthcareParadigm_Insight", "SupplyChain_SN_Proxy", weight=0.7)
                conscious_cube._update_viz_properties("NewHealthcareParadigm_Insight")
                conscious_cube._update_edge_viz_properties("NewHealthcareParadigm_Insight", "Healthcare_SN_Proxy")
                conscious_cube._update_edge_viz_properties("NewHealthcareParadigm_Insight", "SupplyChain_SN_Proxy")


    print("\n--- PHASE 4: Pattern Detection Monitors and Triggers Optimization ---")
    # Establish a baseline for system health (e.g., overall system activity score)
    baseline_system_activity = np.random.normal(loc=0.5, scale=0.1, size=500) # Simulate normal system activity
    pattern_detector.establish_baseline(baseline_system_activity)

    # Simulate a period of "normal" system activity
    current_system_activity_normal = np.random.normal(loc=0.55, scale=0.08, size=50)
    is_emergent, kl_val, optimization_directive_normal = pattern_detector.detect_emergent_pattern(current_system_activity_normal, kl_threshold=0.1)
    if is_emergent:
        print(f"  System-wide Optimization Triggered: {optimization_directive_normal}")

    # Simulate an "emergent pattern" - e.g., a sudden increase in overall system activity/stress
    print("\n--- SIMULATING EMERGENT SYSTEM STRESS ---")
    emergent_system_activity = np.random.normal(loc=0.8, scale=0.15, size=50) # Higher, more volatile activity
    is_emergent, kl_val, optimization_directive_emergent = pattern_detector.detect_emergent_pattern(emergent_system_activity, kl_threshold=0.1)
    if is_emergent:
        print(f"  System-wide Optimization Triggered: {optimization_directive_emergent}")
        # Conceptual: This directive from Pattern Detection would directly influence the Cube's self-architecture
        # For example, if directive is to "PRIORITIZE RESOURCE ALLOCATION ANALYSIS", the Cube might
        # strengthen connections to resource-related Super Nodes or spawn a "ResourceMonitor_Insight" node.
        if "PRIORITIZE RESOURCE ALLOCATION ANALYSIS" in optimization_directive_emergent:
            conscious_cube.update_node_activity("Finance_SN_Proxy", 1.0) # Highlight finance due to resource concern
            conscious_cube.update_node_activity("SupplyChain_SN_Proxy", 1.0) # Highlight supply chain
            conscious_cube.add_conceptual_node("ResourceMonitor_Insight", initial_strength=0.8)
            conscious_cube.graph.add_edge("ResourceMonitor_Insight", "Finance_SN_Proxy", weight=0.9)
            conscious_cube.graph.add_edge("ResourceMonitor_Insight", "SupplyChain_SN_Proxy", weight=0.9)
            conscious_cube._update_viz_properties("ResourceMonitor_Insight")
            conscious_cube._update_edge_viz_properties("ResourceMonitor_Insight", "Finance_SN_Proxy")
            conscious_cube._update_edge_viz_properties("ResourceMonitor_Insight", "SupplyChain_SN_Proxy")
            print("  Conscious Cube responded to optimization directive by highlighting relevant nodes and creating 'ResourceMonitor_Insight'.")


    print("\n--- FINAL SYSTEM STATUS REPORT ---")
    conscious_cube.report_status()

    # Visualize the Conscious Cube using voxels
    conscious_cube.visualize_voxels()

    # --- Conceptual Visualization of Pattern Detection (using Matplotlib) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(baseline_system_activity, bins=pattern_detector.bin_edges, density=True, alpha=0.7, color='blue', label='Baseline Activity')
    plt.hist(current_system_activity_normal, bins=pattern_detector.bin_edges, density=True, alpha=0.7, color='green', label='Normal Current Activity')
    plt.title('System Activity: Normal State vs. Baseline')
    plt.xlabel('Activity Score')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(baseline_system_activity, bins=pattern_detector.bin_edges, density=True, alpha=0.5, color='blue', label='Baseline Activity')
    plt.hist(emergent_system_activity, bins=pattern_detector.bin_edges, density=True, alpha=0.7, color='red', label='Emergent (Stressed) Activity')
    plt.title('System Activity: Emergent Stress vs. Baseline')
    plt.xlabel('Activity Score')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\n=====================================================")
    print("=== KALEIDOSCOPE AI SYSTEM SIMULATION COMPLETE! ===")
    print("=====================================================")

