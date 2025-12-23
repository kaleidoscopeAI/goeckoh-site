def get_similarity_result(task_id):
    task = celery.AsyncResult(task_id)
    if task.ready():
        return jsonify({'result': task.result})
    else:
        return jsonify({'status': task.status}) #Return status to frontend

