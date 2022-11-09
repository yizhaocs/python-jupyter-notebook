import pandas as pd

def incident_target_column_parsing(raw_data, options):
    feature_attrs = options['feature_attrs']
    if 'Incident Target' not in feature_attrs:
        return raw_data

    feature_attrs.remove('Incident Target')
    feature_attrs.append('incident_target_parsed_hostIpAddr')
    feature_attrs.append('incident_target_parsed_hostName')
    incident_target_parsed = raw_data['Incident Target'].str.split(pat=',', expand=False)

    hostIpAddr_data = []
    hostName_data = []
    for i in range(len(incident_target_parsed)):
        e = incident_target_parsed.iloc[i]
        if isinstance(e, list):
            # print(f'e:{e}')
            for i in range(0, len(e)):
                s = e[i]
                if 'hostIpAddr' in s:
                    hostIpAddr_data.append(s.partition(':')[2])
                elif 'hostName' in s:
                    hostName_data.append(s.partition(':')[2])
        else:
            hostIpAddr_data.append(pd.np.nan)
            hostName_data.append(pd.np.nan)

    hostIpAddr_data_df = pd.DataFrame(hostIpAddr_data, columns=['incident_target_parsed_hostIpAddr'])
    hostName_data_df = pd.DataFrame(hostName_data, columns=['incident_target_parsed_hostName'])

    raw_data = pd.concat([raw_data, hostIpAddr_data_df], axis=1)
    raw_data = pd.concat([raw_data, hostName_data_df], axis=1)
    return raw_data


def attack_technique_column_parsing(raw_data, options):
    feature_attrs = options['feature_attrs']
    if 'Attack Technique' not in feature_attrs:
        return raw_data
    feature_attrs.remove('Attack Technique')
    feature_attrs.append('techniqueid')
    attack_tactic_parsed = raw_data['Attack Technique'].str.split(pat=',', expand=False)
    techniqueid_data = []

    for i in range(len(attack_tactic_parsed)):
        e = attack_tactic_parsed.iloc[i]
        if e and isinstance(e, list):
            techniqueid = None
            for i in range(0, len(e)):
                s = e[i]
                if 'techniqueid' in s:
                    import re

                    techniqueid_tmp = s.partition(':')[2]

                    list_of_char = ['\"', '}', '\]', ' ']
                    pattern = '[' + ''.join(list_of_char) + ']'
                    if techniqueid is None:
                        techniqueid_tmp = re.sub(pattern, '', techniqueid_tmp)
                        techniqueid = techniqueid_tmp
                    else:
                        techniqueid_tmp = re.sub(pattern, '', techniqueid_tmp)
                        techniqueid = techniqueid + ',' + techniqueid_tmp

            techniqueid_data.append(techniqueid)
        else:
            techniqueid_data.append(pd.np.nan)

    techniqueid_data_df = pd.DataFrame(techniqueid_data, columns=['techniqueid'])
    raw_data = pd.concat([raw_data, techniqueid_data_df], axis=1)
    print(f'raw_data.columns:{raw_data.columns}')
    return raw_data


def data_praser(raw_data, options):
    raw_data = incident_target_column_parsing(raw_data, options)
    raw_data = attack_technique_column_parsing(raw_data, options)
    return raw_data