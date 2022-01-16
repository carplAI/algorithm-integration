def send_log(job_id, header, subheader, status, level, direction):
    try:
        dict_log = {"header": header,
                    "subheader": subheader,
                    "status": status,
                    "level": level,
                    "direction": direction}
        print(dict_log)
    except Exception as e:
        send_log(job_id, "", "An Error occurred while sending to AI server", "", 5, False)

