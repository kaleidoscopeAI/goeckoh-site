# ... existing methods ...

def allocate_task_to_cluster(self, task: Dict[str, Any]) -> bool:
    """
    Allocates a task to the most suitable cluster.

    Args:
        task (Dict[str, Any]): The task to allocate.

    Returns:
        bool: True if the task was successfully allocated, False otherwise.
    """
    best_cluster_id = self._find_best_cluster_for_task(task)
    if best_cluster_id:
        for node in self.clusters[best_cluster_id]:
            node.process_task(task)
        logging.info(f"Task {task.get('id', 'N/A')} allocated to cluster {best_cluster_id}")
        return True
    else:
        logging.warning(f"No suitable cluster found for task {task.get('id', 'N/A')}")
        return False

def _find_best_cluster_for_task(self, task: Dict[str, Any]) -> Optional[str]:
    """
    Finds the best cluster for a given task based on node capabilities.

    Args:
        task (Dict[str, Any]): The task to allocate.

    Returns:
        Optional[str]: The ID of the best cluster, or None if no suitable cluster is found.
    """
    task_type = task.get("type", "")
    best_cluster_id = None
    best_match_score = 0

    for cluster_id, nodes in self.clusters.items():
        match_score = self._calculate_cluster_match_score(task_type, nodes)
        if match_score > best_match_score:
            best_match_score = match_score
            best_cluster_id = cluster_id

    return best_cluster_id

def _calculate_cluster_match_score(self, task_type: str, cluster_nodes: List[Node]) -> float:
    """
    Calculates a match score for a cluster based on task type and node capabilities.

    Args:
        task_type (str): The type of the task.
        cluster_nodes (List[Node]): List of nodes in the cluster.

    Returns:
        float: The match score for the cluster.
    """
    if not cluster_nodes:
        return 0.0

    match_scores = [
        1.0 if node.can_handle_task(task_type) else 0.0 for node in cluster_nodes
    ]
    return sum(match_scores) / len(cluster_nodes)


