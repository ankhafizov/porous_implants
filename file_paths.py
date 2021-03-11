def get_path(file_id):
    # DIAMOND PDL 05:
    if  file_id == '123493':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123493/full_recon/20200206101055_123493/TiffSaver-tomo'
    elif  file_id == '123494':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123494/full_recon/20200206140043_123494/TiffSaver-tomo'
    elif  file_id == '123495':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123495/full_recon/20200206141126_123495/TiffSaver-tomo'
    elif  file_id == '123496':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123496/full_recon/20200206141825_123496/TiffSaver-tomo'
    elif  file_id == '123497':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123497/full_recon/20200206143333_123497/TiffSaver-tomo'
    elif  file_id == '123498':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123498/full_recon/20200206145231_123498/TiffSaver-tomo'
    elif  file_id == '123499':
        data_folder = '/nfs/synology-tomodata/external_data/tomo/Diamond/I13'+\
                    f'/2020_02/recon/123499/full_recon/20200206150204_123499/TiffSaver-tomo'
    
    return data_folder


def get_benchtop_setup_paths(polimer_name):
    """
    polimer_name = PDLG-5002 or PDL-05
    """
    root_folder = '/home/krivonosov/reconstruction/'
    if polimer_name == "PDL-05":
        folders = {'PDL-05_1 date: 3_18_2020': '71b15235-3181-4aab-a0e0-785bb9b3f8bc',
                    'PDL-05_1 date: 6_9_2020': '3ce34845-3344-412b-aff9-771a47740ee4',
                    'PDL-05_1 date: 6_16_2020': '35fade58-78d1-4171-8d24-149a6f834a82',
                    'PDL-05_1 date: 6_23_2020': 'd95ec2a8-24a5-454d-9b35-a55384510b47',
                    'PDL-05_1 date: 6_30_2020': 'ce1dcc83-b16f-434a-8855-cb298b71f767',

                    'PDL-05_2 date: 3_18_2020': 'e585c9c2-85d2-4e28-9402-6dd15fb586a3',
                    'PDL-05_2 date: 6_9_2020': '9c324d38-0d21-4d03-9b3c-b121441216bc',
                    'PDL-05_2 date: 6_16_2020': 'f320cb2c-e829-47b5-91f9-142d46410134',
                    'PDL-05_2 date: 6_23_2020': '963228e5-3dad-48c3-a438-79f0d297c846',
                    'PDL-05_2 date: 6_30_2020': '71fc3967-f602-40f5-9809-fda22e3cbd73',
        
                    'PDL-05_3 date: 3_17_2020': '123d157a-15c9-4bb1-ada6-5f7bf6a4b81b',
                    'PDL-05_3 date: 6_9_2020': '333e996e-a700-4d4e-81b8-d290d1408b4e',
                    'PDL-05_3 date: 6_16_2020': 'bfefe457-bed4-47c3-b616-d3196cf493ab',
                    'PDL-05_3 date: 6_23_2020': 'bebc103b-9329-4a8b-9947-eab3820f5fed',
                    'PDL-05_3 date: 6_30_2020': 'fb763afa-dcaa-4cb0-8bd8-69a49fd7af2e'
        }
    elif polimer_name == "PDLG-5002":
        folders = {'PDLG-5002_1 date: 3_18_2020': '968ce006-9100-494f-90bd-388c416e403a',
                   'PDLG-5002_1 date: 6_10_2020': 'ec7dc273-749a-400d-8bc3-edb05fbda71b',
                   'PDLG-5002_1 date: 6_17_2020': 'f31557ef-082f-49dc-ab4e-11f15f9c2d2d',
                   'PDLG-5002_1 date: 6_25_2020': '72a57cec-4ed8-4ef3-b388-b5e35f21a5b4',
                   'PDLG-5002_1 date: 7_2_2020': '1a1a4c5d-77ed-49d4-b10f-6d4926cd3dc4',

                   'PDLG-5002_2 date: 3_23_2020': '1c27d542-38c6-4f2a-9b29-cf6b836e58f0',
                   'PDLG-5002_2 date: 6_10_2020': '8222e32e-b912-4662-bdb7-213350d39ecf',
                   'PDLG-5002_2 date: 6_17_2020': '79153edb-128c-4d08-b21b-701cae34c3bd',
                   'PDLG-5002_2 date: 6_25_2020': '8505079e-6bdf-4d94-a68a-d832660f76d4',
                   'PDLG-5002_2 date: 7_2_2020': 'd1f72d54-d1f6-4c40-bd44-d31dfcfdc7b4',

                   'PDLG-5002_3 date: 3_19_2020': '1095977f-c77d-4fdb-a709-9fa6f14fa70e',
                   'PDLG-5002_3 date: 6_10_2020': '98245105-1887-46ef-a258-37053da146cb',
                   'PDLG-5002_3 date: 6_17_2020': 'a4aed226-e179-4acc-8d27-6e002fa8b5d5',
                   'PDLG-5002_3 date: 6_25_2020': '806c797d-6cc3-4f2e-805c-e6df9557f5f7',
                   'PDLG-5002_3 date: 7_2_2020': 'b47b8879-6edf-4085-af5d-36609075ddf8'
        }
    
    paths = {}
    for key, folder_name in folders.items():
        sample_name = folder_name+'.h5'
        paths.update({key: root_folder+folder_name+'/'+sample_name})

    return paths