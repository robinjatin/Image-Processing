from logger import get_logger
import json

log = get_logger(__name__)


class PostProcess:

    @staticmethod
    def create_json(claim_items, line_items, page_number, available_claim_details_columns, available_line_item_columns,
                    payer_name, payer_template_id, claim_status_code):
        if len(claim_items) != len(line_items):
            # log.warn("something is wrong!, check and line items do not align")
            raise
        claims_json = []
        for i in range(len(claim_items)):
            claim_item = claim_items[i]
            claim_item_json = {}
            for j in range(len(claim_item)):
                claim_item_json[available_claim_details_columns[j]] = claim_item[j]
            claim_details_json = {"claimDetails": claim_item_json}
            all_line_items_json = []
            line_number = 1
            for line in line_items[i]:
                line_item_json = {}
                if len(line) != len(available_line_item_columns):
                    # log.error("something is wrong!, line item len is than expected")
                    return None
                else:
                    for j in range(len(line)):
                        if line[j]:
                            if "left" in line[j].keys():
                                line[j].pop("left")
                            if "position" in line[j].keys():
                                line[j].pop("position")
                        line_item_json[available_line_item_columns[j]] = line[j]
                line_items_json = {"lineItem": line_item_json, "lineNumber": line_number}
                all_line_items_json.append(line_items_json)
                line_number = line_number + 1

            claim_details_json["lineItems"] = all_line_items_json
            claim_details_json["pageNumber"] = page_number
            claim_details_json['payerName'] = payer_name
            claim_details_json['payerId'] = ""
            claim_details_json['payerTemplateId'] = payer_template_id
            claim_details_json['claimStatusCode'] = claim_status_code
            claims_json.append(claim_details_json)
        return claims_json

    @staticmethod
    def convert_to_json(blocks, columns, arr):
        json_extract = {}
        for index in range(len(blocks)):
            item = arr[index]
            json_extract[columns[index]] = item
        return json_extract

    @staticmethod
    def post_process(data, filename):
        for key in data:
            if key == 'Billing_Current_Date':
                value = data[key]
                processed_value = ''
                flag = True
                for items in value:
                    if items.isdigit() and flag:
                        processed_value = processed_value + ' '
                        flag = False
                    if items == '-':
                        processed_value = processed_value + ' '
                        flag = True
                    processed_value = processed_value + items
                    if items == '-':
                        processed_value = processed_value + ' '
                data[key] = [processed_value]

        image_type = filename[-4:]
        image_name = filename.replace(image_type, "")
        with open(str(image_name) + ".json", "w") as write_file:
            json.dump(data, write_file, indent=4)
        log.info(f"Final JSON file created for file {filename}.json")
